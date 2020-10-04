#!/usr/bin/env python3
# coding=utf-8

"""
Seq2Seq
Derived from run_ner.py, this file makes a seq2seq model using transformers and trains it.
This file handles a very large amount of arguments using argparse, which appears to be the standard way
within transformers to specify an experiment.
The training and testing data format is defined in utils_seq2seq.py.
"""

from transformers import EncoderDecoderModel, BertTokenizer, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import islice
from keras.preprocessing.sequence import pad_sequences
import copy
import argparse
from tqdm import tqdm
import re

from os import walk, path
import os

from IPython.display import clear_output


def load_file(file):
    with open(file, encoding='UTF-8') as f:
        for line in f:
            acronym, expanded_form, begin, end, sentence = line.split('\t')
            yield(acronym, expanded_form, begin, end, sentence)

def train_batch_generator(generator, tokenizer=None, dataset=None, batch_size=16, tokens_before_and_after=15):

    batch = list(islice(generator, batch_size))

    while batch:

        acronyms = [point[0] for point in batch]
        expanded_forms = [point[1] for point in batch]
        begins = [point[2] for point in batch]
        ends = [point[3] for point in batch]
        sentences = [point[4] for point in batch]

        batch_tokens = convert_input_tokens(acronyms, begins, ends, sentences, tokenizer, tokens_before_and_after)

        input_elem = tokenizer.batch_encode_plus(batch_tokens,
                                                 max_length=tokens_before_and_after*2 + 3,
                                                 is_pretokenized=True,
                                                 pad_to_max_length=True,
                                                 return_tensors="pt",)
        target_elem = tokenizer.batch_encode_plus(expanded_forms,
                                                 max_length=tokens_before_and_after*2 + 3,
                                                 pad_to_max_length=True,
                                                 return_tensors="pt",)
        yield (input_elem["input_ids"],
               input_elem["attention_mask"],
               target_elem["input_ids"],
               target_elem["attention_mask"],
               target_elem["input_ids"])

        batch = list(islice(generator, batch_size))

def test_batch_generator(generator, tokenizer=None, dataset=None, batch_size=16, tokens_before_and_after=15):

    batch = list(islice(generator, batch_size))

    while batch:

        acronyms = [point[0] for point in batch]
        expanded_forms = [point[1] for point in batch]
        begins = [point[2] for point in batch]
        ends = [point[3] for point in batch]
        sentences = [point[4] for point in batch]

        batch_tokens = convert_input_tokens(acronyms, begins, ends, sentences, tokenizer, tokens_before_and_after)

        target_batch = [tokenizer.tokenize(sent, add_special_tokens=True) for sent in expanded_forms]
        lm_labels = copy.deepcopy(target_batch)
        [sent.insert(0, "[PAD]") for sent in lm_labels]
        target_batch = [tokenizer.convert_tokens_to_ids(x) for x in target_batch]
        lm_labels = [tokenizer.convert_tokens_to_ids(x) for x in lm_labels]

        input_elem = tokenizer.batch_encode_plus(batch_tokens,
                                                 max_length=tokens_before_and_after*2 + 3,
                                                 pad_to_max_length=True,
                                                 return_tensors="pt",)


        attention_masks_decode = [[float(i>0) for i in seq] for seq in target_batch]
        yield (torch.tensor(input_elem["input_ids"], dtype=torch.long),
               torch.tensor(input_elem["attention_mask"]),
               torch.tensor(target_batch, dtype=torch.long),
               torch.tensor(attention_masks_decode),
               torch.tensor(lm_labels))

        batch = list(islice(generator, batch_size))

def batch_loader(tokenizer, data_file, step='test', batch_size=16, train=False):
    print('Reading examples : ' + step)
    input_data_loader = {}

    generator = load_file(data_file)

    if train:
        return train_batch_generator(generator, tokenizer, step, batch_size)
    else:
        return test_batch_generator(generator, tokenizer, step, batch_size)

def convert_input_tokens(acronyms, begins, ends, sentences, tokenizer, tokens_before_and_after):
    batch_tokens = []
    for i in range (len(acronyms)):

        input_text = sentences[i]
        begin = begins[i]
        end = ends[i]
        acronym = acronyms[i]

        text_before_acronym = input_text[:int(begin)]
        text_after_acronym = input_text[int(end):]
        tokens_before_acronym = tokenizer.tokenize(text_before_acronym)
        tokens_after_acronym = tokenizer.tokenize(text_after_acronym)

        # Remove the splitting of words that happens during tokenization
        tokens_to_remove = []
        for i in range(len(tokens_before_acronym)):
            if tokens_before_acronym[i].startswith('##'):
                tokens_before_acronym[i-1] = tokens_before_acronym[i-1] + tokens_before_acronym[i].replace('##', '')
                tokens_to_remove.append(i)

        tokens_to_remove.reverse()
        for token_ind in tokens_to_remove:
            del tokens_before_acronym[token_ind]

        tokens_to_remove = []
        for i in range(len(tokens_after_acronym)):
            if tokens_after_acronym[i].startswith('##'):
                tokens_after_acronym[i-1] = tokens_after_acronym[i-1] + tokens_after_acronym[i].replace('##', '')
                tokens_to_remove.append(i)

        tokens_to_remove.reverse()
        for token_ind in tokens_to_remove:
            del tokens_after_acronym[token_ind]

        # Add pad tokens so we have a constant length.
        if len(tokens_before_acronym) > tokens_before_and_after:
            tokens_before_acronym = tokens_before_acronym[-1*tokens_before_and_after:]
        else:
            tokens_before_acronym = [tokenizer.pad_token] * (tokens_before_and_after - len(tokens_before_acronym)) + tokens_before_acronym

        if len(tokens_after_acronym) > tokens_before_and_after:
            tokens_after_acronym = tokens_after_acronym[:tokens_before_and_after]
        else:
            tokens_after_acronym = tokens_after_acronym + [tokenizer.pad_token] * (tokens_before_and_after - len(tokens_after_acronym))

        # Final tokens are the modified tokens before the acronym, followed by the acronym (split into individual letters)
        # then the modified tokens after the acronym.
        batch_tokens.append(tokens_before_acronym + [c for c in acronym] + tokens_after_acronym)
    return batch_tokens

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )


    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")


    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument('--model_load_path', default=None)
    parser.add_argument('--model_save_path', default=None)
    parser.add_argument('--evaluations_file', default=None)

    args = parser.parse_args()


    # New example based on https://colab.research.google.com/drive/1uVP09ynQ1QUmSE2sjEysHjMfKgo4ssb7?usp=sharing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE: ' + str(device))

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    if args.model_load_path:
        model = torch.load(args.model_load_path)
    else:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'bert-base-cased')
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6)


    model.train()
    train_loss_set = []
    train_loss = 0
    save_step = 500

    if args.do_train:
        for epoch in range(int(args.num_train_epochs)):
            batches = tqdm(batch_loader(tokenizer, args.data_file, step='train', batch_size=args.train_batch_size, train=True), desc='Training')
            for step, batch in enumerate(batches):
                batch = tuple(t.to(device) for t in batch)
                input_ids_encode, attention_mask_encode, input_ids_decode, attention_mask_decode, lm_labels = batch
                optimizer.zero_grad()
                model.zero_grad()

                loss, outputs = model(input_ids=input_ids_encode,
                                  decoder_input_ids=input_ids_decode,
                                  attention_mask = attention_mask_encode,
                                  decoder_attention_mask = attention_mask_decode,
                                  lm_labels=lm_labels)[:2]

                train_loss_set.append(loss.item())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()




    if args.do_eval:
        out = open(args.evaluations_file, 'w')
        print('STARTING EVALUATION')
        model.eval()

        test_batches = tqdm(batch_loader(tokenizer, args.data_file, step='test', batch_size=1, train=False), desc='Evaluating')
        for step, batch in enumerate(test_batches):
            batch = tuple(t.to(device) for t in batch)
            input_ids_encode, attention_mask_encode, input_ids_decode, attention_mask_decode, lm_labels = batch
            with torch.no_grad():
                generated = model.generate(input_ids_encode, attention_mask = attention_mask_encode, decoder_start_token_id=model.config.decoder.pad_token_id,
                                        do_sample=True,
                                        max_length=10,
                                        top_k=200,
                                        top_p=0.75,
                                        num_return_sequences=1,
                                        #num_beams=5,
                                        #no_repeat_ngram_size=2,
                                    )
                for i in range(len(generated)):
                    out.write(f'Generated {i}: {tokenizer.decode(generated[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)}' + '\n')

                out.write('Expected: ' + ' '.join([tokenizer.decode(elem, skip_special_tokens=True, clean_up_tokenization_spaces=True) for elem in input_ids_decode]) + '\n')
                out.write('Lm Labels: ' + ' '.join([tokenizer.decode(elem, skip_special_tokens=True, clean_up_tokenization_spaces=True) for elem in lm_labels]) + '\n')
                out.write('Input: ' + ' '.join([tokenizer.decode(elem, skip_special_tokens=True, clean_up_tokenization_spaces=True) for elem in input_ids_encode]) + '\n\n')

        out.close()

    if args.model_save_path:
        torch.save(model, args.model_save_path)

if __name__ == "__main__":
    main()
