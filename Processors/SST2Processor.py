import os

from .DataProcessor import DataProcessor
from Utils.Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors

class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

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
        return ["0", "1"]

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


def load_data(data_dir, tokenizer, processor, max_length, batch_size, data_type):

    label_list = processor.get_labels()

    if data_type == "train":
        examples = processor.get_train_examples(data_dir)
    elif data_type == "dev":
        examples = processor.get_dev_examples(data_dir)
    elif data_type == "test":
        examples = processor.get_test_examples(data_dir)
    else:
        raise RuntimeError("should be train or dev or test")
    
    features = convert_examples_to_features(
        examples, label_list, max_length, tokenizer)
    dataloader = convert_features_to_tensors(features, batch_size)

    return dataloader, len(examples)
