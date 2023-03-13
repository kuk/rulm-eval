
import os
import sys
import csv
import json
import random
from time import sleep
from pathlib import Path
from collections import Counter

import requests

from tqdm import tqdm as log_progress


########
#
#   CONST
#
#####

TERRA = 'terra'
DANETQA = 'danetqa'
PARUS = 'parus'

OPENAI_TOKEN = os.getenv('OPENAI_TOKEN')

TEXT_DAVINCHI_003 = 'text-davinci-003'
CODE_DAVINCHI_002 = 'code-davinci-002'
CODE_CUSHMAN_001 = 'code-cushman-001'
GPT_35_TURBO_0301 = 'gpt-3.5-turbo-0301'


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


#######
#
#  TERRA
#
#####


# {'premise': '"По словам россиянина, перед ним стояла задача - финишировать впереди ""Форс Индии"". ""Мы начали гонку на покрышках средней жесткости. И я старался отстоять свою позицию на старте, так как все в основном были на мягких шинах""."',
#  'hypothesis': 'Соперники выступали преимущественно на мягких шинах.',
#  'label': 'entailment',
#  'idx': 104}


TERRA_PROMPT = '''Does Premise entail Hypothesis? Choose Yes or No.
Keep it short, choose most probable.

Premise: Трижды он был привлечён судебным приставом к административной ответственности по ст. 17.15 КоАП РФ за неисполнение содержащихся в исполнительном документе требований неимущественного характера. Так как срок для добровольного исполнения истёк, пристрой снесли принудительно.
Hypothesis: Пристрой был снесен.
Entail: Yes

Premise: Для молодого организма это не прошло бесследно. Резкое токсическое воздействие этанола привело к смерти парня. Его тело обнаружила бабушка, которая вернулась на следующий день.
Hypothesis: Молодой организм стал сильнее от этанола.
Entail: No

Premise: {premise}
Hypothesis: {hypothesis}
Entail: '''


def terra_prompt(item, template=TERRA_PROMPT):
    return template.format(
        premise=item['premise'],
        hypothesis=item['hypothesis']
    )


def norm_response_mapping(response, pattern_labels):
    labels = []
    for pattern, label in pattern_labels.items():
        if pattern in response:
            labels.append(label)

    if len(labels) == 1:
        return labels[0]


def norm_terra_response(response):
    return norm_response_mapping(response, {
        'Yes': 'entailment',
        'No': 'not_entailment'
    })


######
#
#   DANETQA
#
######


# {'question': 'Есть ли вода на марсе?',
#  'passage': 'Гидросфера Марса — это совокупность водных запасов планеты Марс, представленная водным льдом в полярных шапках Марса, льдом над поверхностью, сезонными ручьями из жидкой воды и возможными резервуарами жидкой воды и водных растворов солей в верхних слоях литосферы Марса. Гидросфера ... е шапки Марса, так как предполагалось, что они могут состоять из водного льда по аналогии с Антарктидой или Гренландией на Земле, однако высказывалась и гипотеза, что это твёрдый диоксид углерода.',
#  'label': True,


DANETQA_PROMPT = '''Given Passage answer the Question.
Keep answer short, choose Yes or No. Choose most probable.

Passage: Пётр Моисеевич Миронов  — красноармеец Рабоче-крестьянской Красной Армии, участник Великой Отечественной войны, Герой Советского Союза . Пётр Миронов родился в 1904 году в деревне Утринка . После окончания шести классов школы проживал в Москве, работал в сфере общепита. В июне 1941 года Миронов был призван на службу в Рабоче-крестьянскую Красную Армию. С июля 1942 года — на фронтах Великой Отечественной войны.
Question: Был ли миронов в армии?
Answer: Yes

Passage: Брюс Ли  — гонконгский и американский киноактёр, режиссёр, сценарист, продюсер, популяризатор и реформатор в области китайских боевых искусств, мастер боевых искусств, постановщик боевых сцен и философ, основоположник стиля Джит Кун-До. Брюс Ли начал сниматься в кино с детства. Его детское имя — Ли Сяолун , взрослое имя — Ли Чжэньфань .
Question: Правда ли что брюс ли не был бойцом?
Answer: No

Passage: {passage}
Question: {question}
Answer: '''


def danetqa_prompt(item, template=DANETQA_PROMPT):
    return template.format(
        passage=item['passage'],
        question=item['question']
    )


def norm_danetqa_response(response):
    return norm_response_mapping(response, {
        'Yes': True,
        'No': False
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
#  'label': 1,
#  'id': 96}


PARUS_PROMPT_QUESTIONS = {
    'effect': 'Что случилось в результате?',
    'cause': 'Что было причиной?',
}

PARUS_PROMPT = '''Given Premise answer the Question. Choose A or B.
In case not enough information choose most probable.

Premise: Я прибралась дома.
Question: Что было причиной?
A: Я была завалена работой.
B: Я ждала друзей.
Answer: B

Premise: Политик был признан виновным в мошенничестве.
Question: Что случилось в результате?
A: Он был отстранён от должности.
B: Он начал кампанию за переизбрание.
Answer: A

Premise: {premise}
Question: {question}
A: {choice1}
B: {choice2}
Answer: '''

def parus_prompt(item, template=PARUS_PROMPT):
    return template.format(
        premise=item['premise'],
        question=PARUS_PROMPT_QUESTIONS[item['question']],
        choice1=item['choice1'],
        choice2=item['choice2'],
    )


def norm_parus_response(response):
    return norm_response_mapping(response, {
        'A': 0,
        'B': 1
    })


#####
#
#   PROMPTS, NORM RESP
#
###


TASK_PROMPTS = {
    TERRA: terra_prompt,
    DANETQA: danetqa_prompt,
    PARUS: parus_prompt,
}

NORM_RESPONSES = {
    TERRA: norm_terra_response,
    DANETQA: norm_danetqa_response,
    PARUS: norm_parus_response,
}


######
#
#   SCORE
#
#####


def acc_score(id_targets, id_preds):
    total, accurate, skip = 0, 0, 0
    for id in id_targets.keys() & id_preds.keys():
        pred = id_preds.get(id)
        if pred is None:
            skip += 1
            continue

        total += 1
        accurate += id_targets.get(id) == pred
            
    return accurate / total, skip


########
#
#   OPENAI
#
######


# https://platform.openai.com/docs/api-reference/completions/create
# https://platform.openai.com/docs/api-reference/chat/create


def parse_openai_stream(lines):
    for line in lines:
        if line.startswith(b'data: '):
            line = line[len('data: '):]
            if line == b'[DONE]':
                break
            yield json.loads(line)


def post_openai_stream(url, payload, token):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    with requests.post(
        url,
        json=payload,
        headers=headers,
        stream=True
    ) as response:
        response.raise_for_status()
        lines = response.iter_lines()
        yield from parse_openai_stream(lines)


def openai_completions_stream(
        prompt,
        model=TEXT_DAVINCHI_003, max_tokens=128,
        temperature=0, top_p=1, stop=None,
        token=OPENAI_TOKEN
):
    items = post_openai_stream(
        'https://api.openai.com/v1/completions',
        {
            'prompt': prompt,
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop,
            'stream': True
        },
        token
    )
    for item in items:
        yield item['choices'][0]['text']


def openai_chat_completions_stream(
        prompt,
        model=GPT_35_TURBO_0301, max_tokens=128,
        temperature=0, top_p=1, stop=None,
        token=OPENAI_TOKEN
):
    items = post_openai_stream(
        'https://api.openai.com/v1/chat/completions',
        {
            'messages': [{'role': 'user', 'content': prompt}],
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop,
            'stream': True
        },
        token
    )
    for item in items:
        content = item['choices'][0]['delta'].get('content')
        if content:
            yield content


def join_print_tokens(tokens):
    buffer = []

    for token in tokens:
        buffer.append(token)
        sys.stdout.write(token)

    sys.stdout.flush()
    return ''.join(buffer)


def join_tokens(tokens):
    return ''.join(tokens)
