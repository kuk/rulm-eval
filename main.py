
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


TEST = 'test'
FEW_SHOT = 'few_shot'

DANETQA = 'danetqa'
TERRA = 'terra'
PARUS = 'parus'
RWSD = 'rwsd'
RUSSE = 'russe'
RUCOLA = 'rucola'
TASKS = [
    TERRA,
    DANETQA,
    PARUS,
    RWSD,
    RUSSE,
    RUCOLA
]

OPENAI_TOKEN = os.getenv('OPENAI_TOKEN')

OPENAI_DAVINCI_003 = 'openai_davinci_003'
OPENAI_TURBO = 'openai_turbo'

SAIGA_7B = 'saiga_7b'


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


#####
#
#   TASK
#
######


def load_task(name, dir=Path('tasks')):
    path = dir / name / 'test.jsonl'
    lines = read_lines(path)
    test_items = list(parse_jsonl(lines))

    path = dir / name / 'dev.jsonl'
    lines = read_lines(path)
    dev_items = list(parse_jsonl(lines))

    return test_items, dev_items


########
#
#   FEW SHOT
#
#####


def sample_stratify_target(dev_items, k=1):
    target_items = defaultdict(list)
    for item in dev_items:
        target_items[item['target']].append(item)

    for target, items in target_items.items():
        yield from random.sample(items, k)


#######
#
#  PROMPT
#
#######


def task_prompt(task_item_text, instruction, test_item, few_shot_items=(), sep='---'):
    test_item_text = task_item_text(test_item, TEST)
    few_shot_item_texts = [task_item_text(_, FEW_SHOT) for _ in few_shot_items]
    items_text = f'\n{sep}\n'.join(few_shot_item_texts + [test_item_text])
    return f'''{instruction}

{items_text}'''


#####
#
#   EVAL ITEM
#
#####


def init_eval_item(test_item):
    return {
        'id': test_item['id'],
        'prompt': None,
        'output': None,
        'pred': None
    }


def init_eval_items(test_items):
    return [init_eval_item(_) for _ in test_items]


#######
#
#  TERRA
#
#####


# {'premise': '"По словам россиянина, перед ним стояла задача - финишировать впереди ""Форс Индии"". ""Мы начали гонку на покрышках средней жесткости. И я старался отстоять свою позицию на старте, так как все в основном были на мягких шинах""."',
#  'hypothesis': 'Соперники выступали преимущественно на мягких шинах.',
#  'target': 'entailment',
#  'idx': 104}


def target_text(target, split, mapping):
    if split == TEST:
        return ''
    elif split == FEW_SHOT:
        return mapping[target]


def match_output_pred(output, mapping):
    values = set()
    for pattern, value in mapping.items():
        if re.search(pattern, output):
            values.add(value)

    if len(values) == 1:
        return values.pop()


TERRA_INSTRUCTION = 'Прочитай текст, проверь верно ли утверждение. Ответь "да" или "нет.'


def terra_item_text(item, split=FEW_SHOT):
    target = item['target']
    answer = target_text(target, split, {
        'entailment': 'Да',
        'not_entailment': 'Нет',
    })
    premise = item['premise']
    hypothesis = item['hypothesis']
    return f'''Текст: {premise}
Утверждение: {hypothesis}
Верно: {answer}'''


def terra_prompt(test_item, few_shot_items=()):
    return task_prompt(
        terra_item_text,
        TERRA_INSTRUCTION,
        test_item, few_shot_items
    )


def terra_output_pred(output):
    return match_output_pred(output, {
        'Утверждения верны': 'entailment',
        'Утверждения неверны': 'not_entailment',
        'Да': 'entailment',
        'Нет': 'not_entailment'
    })


######
#
#   DANETQA
#
######


# {'question': 'Есть ли вода на марсе?',
#  'passage': 'Гидросфера Марса — это совокупность водных запасов планеты Марс, представленная водным льдом в полярных шапках Марса, льдом над поверхностью, сезонными ручьями из жидкой воды и возможными резервуарами жидкой воды и водных растворов солей в верхних слоях литосферы Марса. Гидросфера ... е шапки Марса, так как предполагалось, что они могут состоять из водного льда по аналогии с Антарктидой или Гренландией на Земле, однако высказывалась и гипотеза, что это твёрдый диоксид углерода.',
#  'target': True,


DANETQA_INSTRUCTION = 'Прочитай текст и ответь на вопрос. Ответь коротко "да" или "нет".'


def danetqa_item_text(item, split=FEW_SHOT):
    target = item['target']
    answer = target_text(target, split, {
        True: 'Да',
        False: 'Нет'
    })
    passage = item['passage']
    question = item['question']
    return f'''Текст: {passage}
Вопрос: {question}
Ответ: {answer}'''


def danetqa_prompt(test_item, few_shot_items=()):
    return task_prompt(
        danetqa_item_text,
        DANETQA_INSTRUCTION,
        test_item, few_shot_items
    )


def danetqa_output_pred(output):
    return match_output_pred(output, {
        'Да': True,
        'Нет': False
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


PARUS_PROMPT_QUESTIONS = {
    'effect': 'Что случилось в результате?',
    'cause': 'Что было причиной?',
}

PARUS_INSTRUCTION = 'Ответь на вопрос про причинно-следственную связь. Выбери вариант ответа A или B'


def parus_item_text(item, split=FEW_SHOT):
    target = item['target']
    answer = target_text(target, split, {
        0: 'A',
        1: 'B'
    })
    premise = item['premise']
    question = PARUS_PROMPT_QUESTIONS[item['question']]
    choice1 = item['choice1']
    choice2 = item['choice2']
    return f'''Текст: {premise}
Вопрос: {question}
A. {choice1}
B. {choice2}
Ответ: {answer}'''


def parus_prompt(test_item, few_shot_items=()):
    return task_prompt(
        parus_item_text,
        PARUS_INSTRUCTION,
        test_item, few_shot_items
    )


def parus_output_pred(output):
    return match_output_pred(output, {
        'Выбор A или B зависит': None,
        'Выбор A': 0,
        'Выбор: A': 0,
        'Выбор: B': 1
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


RWSD_INSTRUCTION = 'Прочитай текст и ответь на вопрос про кореференцию. Ответь "да" или "нет".'


def rwsd_item_text(item, split=FEW_SHOT):
    target = item['target']
    answer = target_text(target, split, {
        True: 'Да',
        False: 'Нет'
    })
    text = item['text']
    a = item['target_']['span1_text']
    b = item['target_']['span2_text']
    return f'''Текст: {text}
Вопрос: Фраза "{b}" ссылается на "{a}"?
Ответ: {answer}'''


def rwsd_prompt(test_item, few_shot_items=()):
    return task_prompt(
        rwsd_item_text,
        RWSD_INSTRUCTION,
        test_item, few_shot_items
    )


def rwsd_output_pred(output):
    return match_output_pred(output, {
        'Да': True,
        'Нет': False
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


RUSSE_INSTRUCTION = 'Ответь на вопрос про значение слова в контексте. Ответь коротко: "да" или "нет".'


def russe_item_text(item, split=FEW_SHOT):
    target = item['target']
    answer = target_text(target, split, {
        True: 'Да',
        False: 'Нет'
    })
    word = item['word']
    a = item['sentence1']
    b = item['sentence2']
    return f'''A: {a}
B: {b}
Вопрос: Слово "{word}" имеет одинаковое значение в A и B?
Ответ: {answer}'''


def russe_prompt(test_item, few_shot_items=()):
    return task_prompt(
        russe_item_text,
        RUSSE_INSTRUCTION,
        test_item, few_shot_items
    )


def russe_output_pred(output):
    return match_output_pred(output, {
        'Да': True,
        'Нет': False
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


RUCOLA_INSTRUCTION = 'Предложение корректное или нет? Проверь синтаксис, семантику и морфологию. Ответь коротко: "да" или "нет".'


def rucola_item_text(item, split=FEW_SHOT):
    target = item['target']
    answer = target_text(target, split, {
        '1': 'Да',
        '0': 'Нет'
    })
    sentence = item['sentence']
    return f'''Предложение: {sentence}
Корректное: {answer}'''


def rucola_prompt(test_item, few_shot_items=()):
    return task_prompt(
        rucola_item_text,
        RUCOLA_INSTRUCTION,
        test_item, few_shot_items
    )


def rucola_output_pred(output):
    return match_output_pred(output, {
        'Да': '1',
        'Нет': '0',
        'Некорректное предложение': '0',
    })


######
#
#   SCORE
#
#####


def acc_score(id_targets, id_preds):
    total, correct = 0, 0
    for id in id_targets.keys() & id_preds.keys():
        pred = id_preds[id]
        target = id_targets[id]
        if pred is not None:
            total += 1
            correct += (pred == target)
            
    return correct, total


########
#
#   OPENAI
#
######


# https://platform.openai.com/docs/api-reference/completions/create
# https://platform.openai.com/docs/api-reference/chat/create


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
    response.raise_for_status()
    return response.json()


def openai_completions(
        prompt,
        model='text-davinci-003', max_tokens=128,
        temperature=0, top_p=1, stop=None,
        token=OPENAI_TOKEN        
):
    data = post_openai(
        'https://api.openai.com/v1/completions',
        {
            'prompt': prompt,
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop,
        },
        token
    )
    return data['choices'][0]['text']


def openai_chat_completions(
        prompt,
        model='gpt-35-turbo', max_tokens=128,
        temperature=0, top_p=1, stop=None,
        token=OPENAI_TOKEN
):
    data = post_openai(
        'https://api.openai.com/v1/chat/completions',
        {
            'messages': [{'role': 'user', 'content': prompt}],
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


def rulm_complete_stream(prompt, model='saiga-7b-q4', max_tokens=128, temperature=0.2):
    response = requests.post(
        f'{RULM_URL}/complete',
        json={
            'prompt': prompt,
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


def show_rulm_stream(items):
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


def rulm_complete(prompt, **kwargs):
    items = rulm_complete_stream(prompt, **kwargs)
    buffer = []
    for item in items:
        if item.get('text'):
            buffer.append(item.get('text'))
    return ''.join(buffer)


#######
#
#   MAP
#
######


def rulm_complete_eval_item(item, **kwargs):
    try:
        item['output'] = rulm_complete(item['prompt'], **kwargs)
    except RulmError as error:
        item['error'] = str(error)
    return item
    

def rulm_map_complete_eval_items(items, max_workers=6, **kwargs):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(partial(rulm_complete_eval_item, **kwargs), items)
