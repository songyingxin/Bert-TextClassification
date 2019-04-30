import os

from .DataProcessor import DataProcessor
from Utils.Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors

class NewsProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['房产', '科技', '财经', '游戏', '娱乐', '时尚', '时政', '家居', '教育', '体育']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


def cnews_data(data_dir, tokenizer, processor, max_length, train_batch_size, dev_batch_size, test_batch_size):

    train_examples = processor.get_train_examples(data_dir)
    dev_examples = processor.get_dev_examples(data_dir)
    test_examples = processor.get_test_examples(data_dir)
    label_list = processor.get_labels()

    train_features = convert_examples_to_features(
        train_examples, label_list, max_length, tokenizer)
    dev_features = convert_examples_to_features(
        dev_examples, label_list, max_length, tokenizer)
    test_features = convert_examples_to_features(
        test_examples, label_list, max_length, tokenizer)

    train_dataloader = convert_features_to_tensors(
        train_features, train_batch_size)
    dev_dataloader = convert_features_to_tensors(dev_features, dev_batch_size)
    test_dataloader = convert_features_to_tensors(
        test_features, test_batch_size)

    return train_dataloader, dev_dataloader,  test_dataloader, label_list, len(train_examples)
