# coding=utf-8

import os
import csv

from .DataProcessor import DataProcessor
from Utils.Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors


class SMSProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_sms(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_sms(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_sms(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 2:
                print(line)
                break
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def _read_sms(self, filename):

        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader((line.replace('\0', '')
                             for line in f), delimiter='\t')
            lines = []

            for line in reader:
                lines.append(line)
            
            return lines
