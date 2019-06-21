# coding=utf-8

import os

from .MultiSentenceProcessor import MultiSentenceProcessor
from Utils.MultiSentences_utils import InputExample, convert_examples_to_features, convert_features_to_tensors


class NewsProcessor(MultiSentenceProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir, max_sentence_num):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train", max_sentence_num)

    def get_dev_examples(self, data_dir, max_sentence_num):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev", max_sentence_num)

    def get_test_examples(self, data_dir, max_sentence_num):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test", max_sentence_num)

    def get_labels(self):
        """See base class."""
        return [u'房产', u'科技', u'财经', u'游戏', u'娱乐', u'时尚', u'时政', u'家居', u'教育', u'体育']

    def _create_examples(self, lines, set_type, max_sentence_num):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
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
