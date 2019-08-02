import os
import sys
import json


from .MultiSentences_utils import InputExample, convert_examples_to_features, convert_features_to_tensors


def read_json(filename):
    """Reads json file: [{}, {}]"""
    with open(input_file, "r", encoding='utf-8') as f:
        examples = json.load(f)
    return examples


def load_json_dataset(filename, set_type):
    """
    文件内格式： [
        {"sentences": [],
        "label": ?}
    ]
    """
    examples = []
    lines = read_json(filename)
    for (i, line) in enumerate(lines):``
        guid = "%s-%s" % (set_type, i)
        texts = line['sentences']
        if len(texts) > max_sentence_num:
            texts = texts[:max_sentence_num]
        else:
            while True:
                if len(texts) >= max_sentence_num:
                    break
                else:
                    texts.append([])
        label = line['label']
        examples.append(
            InputExample(guid=guid, texts=texts, label=label))

    return examples
