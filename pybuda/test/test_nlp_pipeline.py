# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
from pybuda.transformers import pipeline
from test.utils import download_model

def check_results(results, expected, keys):

    assert len(results) == len(expected), "Mismatch is result size"
    for i, result in enumerate(results):
        for key in keys:
            assert result[key] == expected[i][key], f"Failed on index {i} on key {key}"

def test_camembert():

    # Reference: https://huggingface.co/Jean-Baptiste/camembert-ner

    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = download_model(AutoTokenizer.from_pretrained, "Jean-Baptiste/camembert-ner")
    model = download_model(AutoModelForTokenClassification.from_pretrained, "Jean-Baptiste/camembert-ner")

    ##### Process text sample (from wikipedia)

    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    results = nlp("Apple est créée le 1er avril 1976 dans le garage de la maison d'enfance de Steve Jobs à Los Altos en Californie par Steve Jobs, Steve Wozniak et Ronald Wayne14, puis constituée sous forme de société le 3 janvier 1977 à l'origine sous le nom d'Apple Computer, mais pour ses 30 ans et pour refléter la diversification de ses produits, le mot « computer » est retiré le 9 janvier 2015.")

    expected = [{'entity_group': 'ORG',
      'score': 0.9472818374633789,
      'word': 'Apple',
      'start': 0,
      'end': 5},
     {'entity_group': 'PER',
      'score': 0.9838564991950989,
      'word': 'Steve Jobs',
      'start': 74,
      'end': 85},
     {'entity_group': 'LOC',
      'score': 0.9831605950991312,
      'word': 'Los Altos',
      'start': 87,
      'end': 97},
     {'entity_group': 'LOC',
      'score': 0.9834540486335754,
      'word': 'Californie',
      'start': 100,
      'end': 111},
     {'entity_group': 'PER',
      'score': 0.9841555754343668,
      'word': 'Steve Jobs',
      'start': 115,
      'end': 126},
     {'entity_group': 'PER',
      'score': 0.9843501806259155,
      'word': 'Steve Wozniak',
      'start': 127,
      'end': 141},
     {'entity_group': 'PER',
      'score': 0.9841533899307251,
      'word': 'Ronald Wayne',
      'start': 144,
      'end': 157},
     {'entity_group': 'ORG',
      'score': 0.9468960364659628,
      'word': 'Apple Computer',
      'start': 243,
      'end': 257}]

    check_results(results, expected, keys=["entity_group", "word", "start", "end"])


def test_distilroberta_base():

    # Reference: https://huggingface.co/distilroberta-base

    unmasker = pipeline('fill-mask', model='distilroberta-base')

    results = unmasker("Hello I'm a <mask> model.")
    
    expected = [{'score': 0.10418225079774857, 'token': 774, 'token_str': ' role', 'sequence': "Hello I'm a role model."}, {'score': 0.04240003600716591, 'token': 265, 'token_str': ' business', 'sequence': "Hello I'm a business model."}, {'score': 0.03184013068675995, 'token': 2734, 'token_str': ' fashion', 'sequence': "Hello I'm a fashion model."}, {'score': 0.02836352400481701, 'token': 18150, 'token_str': ' freelance', 'sequence': "Hello I'm a freelance model."}, {'score': 0.02524581365287304, 'token': 5704, 'token_str': ' fitness', 'sequence': "Hello I'm a fitness model."}]

    check_results(results, expected, keys=["sequence", "token", "token_str"])

def test_bert_base_uncased():

    # Reference: https://huggingface.co/bert-base-uncased

    # TODO: this one doesn't produce correct results

    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    results = unmasker("Hello I'm a [MASK] model.")

    expected = [{'sequence': "[CLS] hello i'm a fashion model. [SEP]",
      'score': 0.1073106899857521,
      'token': 4827,
      'token_str': 'fashion'},
     {'sequence': "[CLS] hello i'm a role model. [SEP]",
      'score': 0.08774490654468536,
      'token': 2535,
      'token_str': 'role'},
     {'sequence': "[CLS] hello i'm a new model. [SEP]",
      'score': 0.05338378623127937,
      'token': 2047,
      'token_str': 'new'},
     {'sequence': "[CLS] hello i'm a super model. [SEP]",
      'score': 0.04667217284440994,
      'token': 3565,
      'token_str': 'super'},
     {'sequence': "[CLS] hello i'm a fine model. [SEP]",
      'score': 0.027095865458250046,
      'token': 2986,
      'token_str': 'fine'}]

    check_results(results, expected, keys=["sequence", "token", "token_str"])


def test_gpt2():

    # Reference: https://huggingface.co/gpt2

    # TODO: this uses model.generate(..) which we need to fall into somehow

    from transformers import set_seed
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    results = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

    print(results)

def test_translation():

    # Fails with 'RuntimeError: Tracing can't be nested' in jit trace

    en_fr_translator = pipeline("translation_en_to_fr", model='t5-base')
    print(en_fr_translator("How old are you?"))
