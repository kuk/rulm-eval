
import re
import os
import sys
import csv
import json
import random
from time import sleep
from pathlib import Path
from collections import (
    Counter,
    defaultdict
)
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import requests

import pandas as pd
from matplotlib import pyplot as plt

from tqdm import tqdm as log_progress


########
#
#   CONST
#
#####

AMBIG = 'ambig'

DANETQA = 'danetqa'
TERRA = 'terra'
PARUS = 'parus'
RWSD = 'rwsd'
RUSSE = 'russe'
MUSERC = 'muserc'
RCB = 'rcb'
RUCOS = 'rucos'
RUCOLA = 'rucola'
TASKS = [
    # QA over text
    DANETQA,
    MUSERC,

    # Reading compr
    RUCOS,

    # NLI
    TERRA,
    RCB,

    # Common sense
    PARUS,
    RUSSE,

    # Lingvo
    RWSD,
    # RUCOLA,
]

OPENAI_TOKEN = os.getenv('OPENAI_TOKEN')

OPENAI_DAVINCI_003 = 'openai_davinci_003'
OPENAI_TURBO = 'openai_turbo'

SAIGA_7B = 'saiga_7b'
SAIGA_13B = 'saiga_13b'
SAIGA_30B = 'saiga_30b'

MODELS = [
    OPENAI_TURBO,
    SAIGA_7B,
    SAIGA_13B,
    SAIGA_30B,
]

######
#
#   RSG LB
#
#####


def parse_rsg_lb_cell(value):
    match = re.match(r'[\d\.\-]+', value)
    if match:
        return float(match.group())
    return value


def parse_rsg_lb(text):
    rows = [
        _.split('\t')
        for _ in text.strip().splitlines()
    ]
    header, body = rows[0], rows[1:]
    header = [_.lower() for _ in header]
    for row in body:
        row = [parse_rsg_lb_cell(_) for _ in row]
        yield dict(zip(header, row))


RSG_LB = list(parse_rsg_lb('''
Rank	Name	Team	Link	Score	LiDiRus	RCB	PARus	MuSeRC	TERRa	RUSSE	RWSD	DaNetQA	RuCoS
1	HUMAN BENCHMARK	AGI NLP		0.811	0.626	0.68 / 0.702	0.982	0.806 / 0.42	0.92	0.805	0.84	0.915	0.93 / 0.89
2	FRED-T5 1.7B finetune	SberDevices		0.762	0.497	0.497 / 0.541	0.842	0.916 / 0.773	0.871	0.823	0.669	0.889	0.9 / 0.902
3	Golden Transformer v2.0	Avengers Ensemble		0.755	0.515	0.384 / 0.534	0.906	0.936 / 0.804	0.877	0.687	0.643	0.911	0.92 / 0.924
4	YaLM p-tune (3.3B frozen + 40k trainable params)	Yandex		0.711	0.364	0.357 / 0.479	0.834	0.892 / 0.707	0.841	0.71	0.669	0.85	0.92 / 0.916
5	FRED-T5 large finetune	SberDevices		0.706	0.389	0.456 / 0.546	0.776	0.887 / 0.678	0.801	0.775	0.669	0.799	0.87 / 0.863
6	RuLeanALBERT	Yandex Research		0.698	0.403	0.361 / 0.413	0.796	0.874 / 0.654	0.812	0.789	0.669	0.76	0.9 / 0.902
7	FRED-T5 1.7B (only encoder 760M) finetune	SberDevices		0.694	0.421	0.311 / 0.441	0.806	0.882 / 0.666	0.831	0.723	0.669	0.735	0.91 / 0.911
8	ruT5-large finetune	SberDevices		0.686	0.32	0.45 / 0.532	0.764	0.855 / 0.608	0.775	0.773	0.669	0.79	0.86 / 0.859
9	ruRoberta-large finetune	SberDevices		0.684	0.343	0.357 / 0.518	0.722	0.861 / 0.63	0.801	0.748	0.669	0.82	0.87 / 0.867
10	Golden Transformer v1.0	Avengers Ensemble		0.679	0.0	0.406 / 0.546	0.908	0.941 / 0.819	0.871	0.587	0.545	0.917	0.92 / 0.924
11	xlm-roberta-large (Facebook) finetune	SberDevices		0.654	0.369	0.328 / 0.457	0.59	0.809 / 0.501	0.798	0.765	0.669	0.757	0.89 / 0.886
12	mdeberta-v3-base (Microsoft) finetune	SberDevices		0.651	0.332	0.27 / 0.489	0.716	0.825 / 0.531	0.783	0.727	0.669	0.708	0.87 / 0.868
13	ruT5-base finetune	Sberdevices		0.635	0.267	0.423 / 0.461	0.636	0.808 / 0.475	0.736	0.707	0.669	0.769	0.85 / 0.847
14	ruBert-large finetune	SberDevices		0.62	0.235	0.356 / 0.5	0.656	0.778 / 0.436	0.704	0.707	0.669	0.773	0.81 / 0.805
15	ruBert-base finetune	SberDevices		0.578	0.224	0.333 / 0.509	0.476	0.742 / 0.399	0.703	0.706	0.669	0.712	0.74 / 0.716
16	YaLM 1.0B few-shot	Yandex		0.577	0.124	0.408 / 0.447	0.766	0.673 / 0.364	0.605	0.587	0.669	0.637	0.86 / 0.859
17	RuGPT3XL few-shot	SberDevices		0.535	0.096	0.302 / 0.418	0.676	0.74 / 0.546	0.573	0.565	0.649	0.59	0.67 / 0.665
18	RuBERT plain	DeepPavlov		0.521	0.191	0.367 / 0.463	0.574	0.711 / 0.324	0.642	0.726	0.669	0.639	0.32 / 0.314
19	SBERT_Large_mt_ru_finetuning	SberDevices		0.514	0.218	0.351 / 0.486	0.498	0.642 / 0.319	0.637	0.657	0.675	0.697	0.35 / 0.347
20	SBERT_Large	SberDevices		0.51	0.209	0.371 / 0.452	0.498	0.646 / 0.327	0.637	0.654	0.662	0.675	0.36 / 0.351
21	RuGPT3Large	SberDevices		0.505	0.231	0.417 / 0.484	0.584	0.729 / 0.333	0.654	0.647	0.636	0.604	0.21 / 0.202
22	RuBERT conversational	DeepPavlov		0.5	0.178	0.452 / 0.484	0.508	0.687 / 0.278	0.64	0.729	0.669	0.606	0.22 / 0.218
23	Multilingual Bert	DeepPavlov		0.495	0.189	0.367 / 0.445	0.528	0.639 / 0.239	0.617	0.69	0.669	0.624	0.29 / 0.29
24	heuristic majority	hse_ling		0.468	0.147	0.4 / 0.438	0.478	0.671 / 0.237	0.549	0.595	0.669	0.642	0.26 / 0.257
25	RuGPT3Medium	SberDevices		0.468	0.01	0.372 / 0.461	0.598	0.706 / 0.308	0.505	0.642	0.669	0.634	0.23 / 0.224
26	RuGPT3Small	SberDevices		0.438	-0.013	0.356 / 0.473	0.562	0.653 / 0.221	0.488	0.57	0.669	0.61	0.21 / 0.204
27	Baseline TF-IDF1.1	AGI NLP		0.434	0.06	0.301 / 0.441	0.486	0.587 / 0.242	0.471	0.57	0.662	0.621	0.26 / 0.252
'''))

RSG_LB_HUMAN = 'HUMAN BENCHMARK'


def rsg_lb_human(items, tasks=TASKS):
    for item in items:
        if item['name'] == RSG_LB_HUMAN:
            for task in tasks:
                score = item.get(task)
                if score:
                    yield task, score


def rsg_lb_sota(items, tasks=TASKS):
    for task in tasks:
        scores = []
        for item in items:
            score = item.get(task)
            if score and item['name'] != RSG_LB_HUMAN:
                scores.append(score)
        if scores:
            yield task, max(scores)


########
#
#  RUCOLA LB
#
#######

# https://rucola-benchmark.com/leaderboard


RUCOLA_LB_HUMAN = 0.84
RUCOLA_LB_SOTA = 0.82


#######
#
#   LINES
#
######


def read_lines(path):
    with open(path) as file:
        for line in file:
            yield line.rstrip('\n')


def write_lines(path, lines):
    with open(path, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def append_lines(path, lines):
    with open(path, 'a') as file:
        for line in lines:
            file.write(line + '\n')


#######
#
#   JSONL
#
#####


def parse_jsonl(lines):
    for line in lines:
        yield json.loads(line)


def format_jsonl(items):
    for item in items:
        yield json.dumps(item, ensure_ascii=False, indent=None)


#######
#
#  CSV
#
######


def read_csv(path):
    with open(path, newline='') as file:
        yield from csv.DictReader(file)


########
#
#   DOTENV
#
######


DOTENV_PATH = '.env'


def parse_dotenv(lines):
    for line in lines:
        if line:
            key, value = line.split('=', 1)
            yield key, value


#######
#
#  TERRA
#
#####


# {'premise': '"По словам россиянина, перед ним стояла задача - финишировать впереди ""Форс Индии"". ""Мы начали гонку на покрышках средней жесткости. И я старался отстоять свою позицию на старте, так как все в основном были на мягких шинах""."',
#  'hypothesis': 'Соперники выступали преимущественно на мягких шинах.',
#  'target': 'entailment',
#  'idx': 104}


def match_pred(text, mapping):
    text = text.strip()

    preds = set()
    for pattern, pred in mapping.items():
        if re.search(pattern, text):
            preds.add(pred)

    if AMBIG in preds:
        return AMBIG

    if len(preds) == 1:
        return preds.pop()


def terra_agent(item, ctx):
    premise = item['premise']
    hypothesis = item['hypothesis'].rstrip('.')
    prompt = f'Дан текст: ```{premise}``` Следует ли из текста что {hypothesis}?'
    
    response = ctx.send(prompt)
    pred = terra_pred(response)

    if pred is None:
        response = ctx.send('Финальный ответ (только "да" или "нет"):')
        pred = terra_pred(response)

    return pred


def terra_pred(text):
    return match_pred(text, {
        r'^Да\b': 'entailment',
        r'^Нет\b': 'not_entailment',
    })


######
#
#   DANETQA
#
######


# {'question': 'Есть ли вода на марсе?',
#  'passage': 'Гидросфера Марса — это совокупность водных запасов планеты Марс, представленная водным льдом в полярных шапках Марса, льдом над поверхностью, сезонными ручьями из жидкой воды и возможными резервуарами жидкой воды и водных растворов солей в верхних слоях литосферы Марса. Гидросфера ... е шапки Марса, так как предполагалось, что они могут состоять из водного льда по аналогии с Антарктидой или Гренландией на Земле, однако высказывалась и гипотеза, что это твёрдый диоксид углерода.',
#  'target': True,


def danetqa_agent(item, ctx):
    passage = item['passage']
    question = item['question']
    prompt = f'''Дан текст: ```{passage}``` Ответь на вопрос по тексту: {question}'''

    response = ctx.send(prompt)
    pred = danetqa_pred(response)

    if pred is None:
        response = ctx.send('Финальный ответ (только "да" или "нет"):')
        pred = danetqa_pred(response)

    return pred


def danetqa_pred(text):
    return match_pred(text, {
        r'^Да\b': True,
        r'^Нет\b': False,
    })


#####
#
#   PARUS
#
#####


# {'premise': 'Я прибралась дома.',
#  'choice1': 'Я была завалена работой.',
#  'choice2': 'Я ждала друзей.',
#  'question': 'cause',
#  'target': 1,
#  'id': 96}


def parus_agent(item, ctx):
    premise = item['premise']
    choice1 = item['choice1'].rstrip('.')
    choice2 = item['choice2'].rstrip('.')

    if item['question'] == 'effect':
        question = 'Что вероятно произошло после этого'
    else:
        question = 'Что вероятно было причиной'

    ctx.send(f'''{premise} {question} 1. "{choice1}" или 2. "{choice2}"? Подробно рассуждай''')
    response = ctx.send('Финальный ответ, наиболее вероятно (только "1." или "2."):')

    return parus_pred(response)


def parus_pred(text):
    return match_pred(text, {
        r'1\.': 0,
        r'2\.': 1,
    })


#####
#
#   RWSD
#
#####


# {'text': 'Матери Артура и Селесты пришли в город, чтобы забрать их. Они очень рады, что их вернули, но они также ругают их, потому что они убежали.',
#  'target': {'span2_index': 8,
#   'span1_index': 0,
#   'span1_text': 'Матери',
#   'span2_text': 'забрать их'},
#  'idx': 190,
#  'target': False}


def rwsd_agent(item, ctx):
    text = item['text']
    a = item['target_']['span1_text']
    b = item['target_']['span2_text']
    ctx.send(f'''Дан текст ```{text}``` На кого или на что ссылается местоимение "{b}" в тексте?''')
    response = ctx.send(f'Местоимение "{b}" ссылается на "{a}"?')
    return rwsd_pred(response)


def rwsd_pred(text):
    return match_pred(text, {
        r'^Да\b': True,
        r'^Нет\b': False,
    })


######
#
#   RUSSE
#
#####


# {'idx': 4107,
#  'word': 'защита',
#  'sentence1': 'Как изменится защита Динамо в новом сезоне?',
#  'sentence2': 'Обе партии протекали на удивление одинаково: в обеих была разыграна..
#  'start1': 14,
#  'end1': 21,
#  'start2': 80,
#  'end2': 87,
#  'target': True,
#  'gold_sense1': 2,
#  'gold_sense2': 2}


def russe_agent(item, ctx):
    word = item['word']
    a = item['sentence1']
    b = item['sentence2']
    prompt = f'В предложениях "{a}" и "{b}" слово "{word}" употребляется в одинаковом значении?'

    response = ctx.send(prompt)
    pred = russe_pred(response)

    if pred is None:
        response = ctx.send('Финальный ответ (только "да" или "нет"):')
        pred = russe_pred(response)

    return pred


def russe_pred(text):
    return match_pred(text, {
        r'^Да\b': True,
        r'^Нет\b': False,
    })


######
#
#  RUCOLA
#
#####


# {'id': '49',
#  'sentence': 'Мне бы хотелось открыться кому-нибудь, но разве здесь есть такие люди, которые бы могли меня понять.',
#  'acceptable': '1',
#  'error_type': '0',
#  'detailed_source': 'Seliverstova'}


def rucola_agent(item, ctx):
    before = item['sentence']
    after = ctx.send(f'Перепиши предложение "{before}" на грамотном русском языке сохраняя смысл')
    return rucola_pred(before, after)


def rucola_pred(before, after):
    after = after.strip('" \n')
    if before == after:
        return '1'
    else:
        return '0'


######
#
#  MUSERC
#
#####


# {
#   "id": 494,
#   "passage": "(1) Пожаловавшийся президенту России Владимиру Путину на отсутствие доходов фермер Джон Кописки решил продать свое хозяйство. (2) Об этом сообщает ТАСС со ссылкой на заявление самого фермера. (3) По его словам, он готов продать ферму во Владимирской области за три миллиарда рублей. (4) Впрочем, если найдется хороший и работящий покупатель, то Джон обещает уступить ему хозяйство по лучшей цене.«Продавать будет сложно. (5) Банку я
#   "question": "Каким было решение Джона Кописки по поводу фермы?",
#   "answer": "Подать ферму Владимиру Путину.",
#   "target": 0
# }


def muserc_agent(item, ctx):
    passage = item['passage']
    question = item['question']
    answer = item['answer']
    ctx.send(f'''Дан текст: ```{passage}``` Ответь на вопрос по тексту: {question}''')

    response = ctx.send(f'Ответ "{answer}" это правильный ответ на вопрос?')
    return muserc_pred(response)
    

def muserc_pred(text):
    return match_pred(text, {
        r'^Да\b': 1,
        r'^Нет\b': 0,
    })


######
#
#  RCB
#
#####


# {
#   "premise": "Я добиваюсь того, чтобы мои голоса не крали. Я добиваюсь того, чтобы люди поняли, что их много, и они вместе.",
#   "hypothesis": "Людей много, и они вместе.",
#   "verb": "понять",
#   "negation": "no_negation",
#   "genre": "interfax",
#   "id": 316,
#   "target": "entailment"
# }


def rcb_agent(item, ctx):
    premise = item['premise']
    hypothesis = item['hypothesis']
    response = ctx.send(f'Дан текст: ```{premise}``` Из текста следует что "{hypothesis}"?')
    pred = rcb_pred(response)

    if pred is None:
        response = ctx.send(f'Финальный ответ (только "да" или "нет"):')
        pred = rcb_pred(response)

    return pred


def rcb_pred(text):
    return match_pred(text, {
        r'^Да\b': 'entailment',
        r'^Нет\b': 'contradiction',
    })


######
#
#  RUCOS
#
#####


# {
#   "id": 63988,
#   "text": "Молдавский телеканал TV7 с 1 мая прекратит трансляцию информационных и аналитических программ российского канала НТВ, на базе которых он строит свою сетку вещания, передает «Интерфакс» со ссылкой на сообщение молдавской телекомпании. Решение отказаться от контента российского телеканала продиктовано желанием TV7 перейти на собственное информационное вещание. На настоящий момент информационные и аналитические передачи НТВ составляют основу сетки молдавского канала. Российские каналы вещают в Молдавии посредством каналов-партнеров. Местные каналы включают в сетку свои передачи, информационные выпуски и рекламу. Каналы из России получают оплату от трансляции или часть доходов от рекламы.\n- В Молдавии запретили трансляцию «России 24»\n- В Молдавии оштрафовали ретрансляторов российского ТВ\n- Молдавия осталась без Первого канала из-за фальшивки",
#   "query": "Размещением рекламы на @placeholder занимается подразделение собственного селлера НТВ — «Алькасар».",
#   "entity": "России 24",
#   "target": false
# }


def rucos_agent(item, ctx):
    text = item['text']
    query = item['query'].replace('@placeholder', '__________')
    entity = item['entity']

    prompt = f'Дан текст: ```{text}``` В вопросе "{query}" пропущено слово "{entity}"?'
    response = ctx.send(prompt)
    pred = rucos_pred(response)

    if pred is None:
        response = ctx.send('Финальный ответ (только "да" или "нет"):')
        pred = rucos_pred(response)

    return pred


def rucos_pred(text):
    return match_pred(text, {
        r'^Да\b': True,
        r'^Нет\b': False,
    })


#####
#
#   TASK DISPATCH
#
####


TASK_AGENTS = {
    TERRA: terra_agent,
    DANETQA: danetqa_agent,
    PARUS: parus_agent,
    RWSD: rwsd_agent,
    RUSSE: russe_agent,
    RUCOLA: rucola_agent,
    MUSERC: muserc_agent,
    RCB: rcb_agent,
    RUCOS: rucos_agent
}


######
#
#   SCORE
#
#####


def acc_score(id_targets, id_preds):
    correct, support, total = 0, 0, 0
    for id in id_targets.keys() & id_preds.keys():
        pred = id_preds[id]
        target = id_targets[id]
        total += 1
        if pred not in (None, AMBIG):
            support += 1
            correct += (pred == target)
            
    return correct, support, total


########
#
#   OPENAI
#
######


# https://platform.openai.com/docs/api-reference/completions/create
# https://platform.openai.com/docs/api-reference/chat/create


class OpenaiError(Exception):
    pass


def post_openai(url, payload, token):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    response = requests.post(
        url,
        json=payload,
        headers=headers,
    )
    if response.status_code != 200:
        raise OpenaiError(response.text)

    return response.json()


def openai_chat_complete(
        messages,
        model='gpt-3.5-turbo', max_tokens=256,
        temperature=0, top_p=1, stop=None,
        token=OPENAI_TOKEN
):
    def assign_roles(messages):
        for index, message in enumerate(messages):
            role = (
                'user' if index % 2 == 0
                else 'assistant'
            )
            yield {
                'role': role,
                'content': message
            }

    data = post_openai(
        'https://api.openai.com/v1/chat/completions',
        {
            'messages': list(assign_roles(messages)),
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop,
        },
        token
    )
    return data['choices'][0]['message']['content']


######
#
#  RULM
#
####


RULM_URL = 'https://api.rulm.alexkuk.ru/v1'


class RulmError(Exception):
    pass


def rulm_models():
    return requests.get(f'{RULM_URL}/models').json()


def rulm_tokenize(text, model='saiga-7b-q4'):
    response = requests.post(
        f'{RULM_URL}/tokenize',
        json={
            'text': text,
            'model': model
        }
    )
    if response.status_code != 200:
        raise RulmError(response.text)
        
    return response.json()


def rulm_chat_complete_stream(
        messages,
        model='saiga-7b-q4',
        max_tokens=256, temperature=0
):
    response = requests.post(
        f'{RULM_URL}/chat_complete',
        json={
            'messages': messages,
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature
        },
        stream=True
    )
    if response.status_code != 200:
        raise RulmError(response.text)

    for line in response.iter_lines():
        item = json.loads(line)
        error = item.get('error')
        if error:
            raise RulmError(error)
        yield item


def rulm_show_stream(items):
    buffer = []
    for item in items:
        text = item.get('text')
        prompt_progress = item.get('prompt_progress')
        if text:
            buffer.append(text)
            print(text, flush=True, end='')
        else:
            print(f'{prompt_progress * 100:.0f}%', flush=True, end=' ')
            if prompt_progress == 1:
                print('\n', flush=True)
    return ''.join(buffer)


def rulm_chat_complete(messages, **kwargs):
    items = rulm_chat_complete_stream(messages, **kwargs)
    buffer = []
    for item in items:
        if item.get('text'):
            buffer.append(item.get('text'))
    return ''.join(buffer)


#######
#
#   RUN AGENT
#
######


class AgentContext:
    Error = None
    MODELS = None

    def __init__(self, model):
        self.model = model
        self.messages = []


MODEL_API_NAMES = {
    SAIGA_7B: 'saiga-7b-q4',
    SAIGA_13B: 'saiga-13b-q4',
    OPENAI_TURBO: 'gpt-3.5-turbo'
}


class RulmAgentContext(AgentContext):
    Error = RulmError

    def send(self, user_message, **kwargs):
        self.messages.append(user_message)
        bot_message = rulm_chat_complete(
            self.messages,
            model=MODEL_API_NAMES[self.model],
            **kwargs
        )
        self.messages.append(bot_message)
        return bot_message


class RulmAgentContextVerbose(RulmAgentContext):
    def send(self, user_message, **kwargs):
        self.messages.append(user_message)
        print(user_message)

        stream = rulm_chat_complete_stream(
            self.messages,
            model=MODEL_API_NAMES[self.model],
            **kwargs
        )
        bot_message = rulm_show_stream(stream)
        self.messages.append(bot_message)
        print()

        return bot_message


class OpenaiAgentContext(AgentContext):
    Error = OpenaiError

    def send(self, user_message, **kwargs):
        self.messages.append(user_message)
        bot_message = openai_chat_complete(
            self.messages,
            model=MODEL_API_NAMES[self.model],
            **kwargs
        )
        self.messages.append(bot_message)
        return bot_message


class OpenaiAgentContextVerbose(OpenaiAgentContext):
    def send(self, user_message, **kwargs):
        self.messages.append(user_message)
        print(user_message)
        print()

        bot_message = openai_chat_complete(
            self.messages,
            model=MODEL_API_NAMES[self.model],
            **kwargs
        )
        self.messages.append(bot_message)
        print(bot_message)
        print()

        return bot_message


def run_agent(agent, test_item, ctx):
    try:
        pred = agent(test_item, ctx)
    except ctx.Error as error:
        print(error, file=sys.stderr)
        pred = None

    return {
        'id': test_item['id'],
        'messages': ctx.messages,
        'pred': pred
    }


def map_agents(agent, test_items, Context, max_workers=6):
    def worker(test_item):
        return run_agent(agent, test_item, ctx=Context())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(worker, test_items)


######
#
#   SHOW HTML
#
#####


def show_html(html):
    from IPython.display import (
        display,
        HTML
    )

    display(HTML(html))
