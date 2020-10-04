#!/usr/bin/env python3
# coding=utf-8

"""
This defines an input example format as well as classes to hold the data.
The logic here was largely derived from that in utils_ner.py though a lot of if statements for tokenization
was moved into the convert_examples_into_features function.
"""
import logging
import os
import copy
from keras.preprocessing.sequence import pad_sequences


logger = logging.getLogger(__name__)


def pairwise(it):
    """
    A function to pairwise retrieve pieces of an iterable
    If given [1,2,3,4], this will return
    (1,2)
    (3,4)
    """
    it = iter(it)
    try:
        while True:
            yield next(it), next(it)
    # in python3.7 this raises a stopiteration that blocks example reading
    except StopIteration:
        pass


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, formatted_words=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            formatted_words: (Optional) list. The formatted version of the input words. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.formatted_words = formatted_words


"""
Example structure:
input:  march 23 2019
output: 3/23/2019
"""


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        formatted_words = []
        for words, formatted_words in pairwise(f):
            words = words.lstrip("input: ").strip()
            formatted_words = formatted_words.lstrip("output: ").strip()
            examples.append(
                InputExample(guid="{}-{}".format(mode, guid_index), words=words, formatted_words=formatted_words,)
            )
            guid_index += 1
    return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self, input_ids, output_ids, input_mask, output_mask, segment_ids, formatted_tokens, lm_labels
    ):
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.input_mask = input_mask
        self.output_mask = output_mask
        self.segment_ids = segment_ids
        self.formatted_tokens = formatted_tokens
        self.lm_labels = lm_labels


def convert_examples_to_features(
    examples, mode, max_seq_length, tokenizer, encoder_model_type, pad_token_label_id=-100,
):
    """
    Loads a data file into a list of `InputBatch`s
    """
    #  `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    cls_token_segment_id = 2 if encoder_model_type in ["xlnet"] else 0
    pad_token_segment_id = 4 if encoder_model_type in ["xlnet"] else 0
    features = []

    cls_token, sep_token, pad_token = (
        tokenizer.cls_token,
        tokenizer.sep_token,
        tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
    )
    sequence_a_segment_id = 0
    mask_padding_with_zero = True

    input_ids = [tokenizer.tokenize(example.words, add_special_tokens=True) for example in examples]
    output_ids = [tokenizer.tokenize(example.formatted_words, add_special_tokens=True) for example in examples]
    lm_labels = copy.deepcopy(output_ids)
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in input_ids]
    output_ids = [tokenizer.convert_tokens_to_ids(x) for x in output_ids]
    lm_labels = [tokenizer.convert_tokens_to_ids(x) for x in lm_labels]
    input_ids = pad_sequences(input_ids, maxlen=max_seq_length, dtype='long', truncating='post', padding='post')
    output_ids = pad_sequences(output_ids, maxlen=max_seq_length, dtype='long', truncating='post', padding='post')
    lm_labels = pad_sequences(lm_labels, maxlen=max_seq_length, dtype='long', truncating='post', padding='post')

    tokens = [tokenizer.tokenize(example.words) for example in examples]
    formatted_tokens = [tokenizer.tokenize(example.formatted_words) for example in examples]
    attention_masks_encode = [[float(i>0) for i in seq] for seq in input_ids]
    attention_masks_decode = [[float(i>0) for i in seq] for seq in output_ids]

    for i in range(len(input_ids)):
        if i < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s", " ".join([str(x) for x in tokens[i]]))
            logger.info("formatted_tokens: %s", " ".join([str(x) for x in formatted_tokens[i]]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids[i]]))
            logger.info("output_ids: %s", " ".join([str(x) for x in output_ids[i]]))
            logger.info("input_mask: %s", " ".join([str(x) for x in attention_masks_encode[i]]))
            logger.info("output_mask: %s", " ".join([str(x) for x in attention_masks_decode[i]]))

        features.append(
            InputFeatures(
                input_ids=input_ids[i],
                output_ids=output_ids[i],
                input_mask=attention_masks_encode[i],
                output_mask=attention_masks_decode[i],
                segment_ids=[],
                formatted_tokens=formatted_tokens[i],
                lm_labels=lm_labels[i]
            )
        )

    return features
