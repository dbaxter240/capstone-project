import preprocess
import nltk
import glob
from tqdm import tqdm
import multiprocessing
from functools import partial
import re

FILES_TO_PARSE = "C:\\Capstone\\Reports\\*[90].txt"
TOKEN_BREAK_SYMBOL = ' '

def extract_train_data(ngrams_to_extract, outfile='extractedTrainData.txt'):
    print(FILES_TO_PARSE)
    ngrams_to_acronyms = {}
    for ngram in ngrams_to_extract:
        acr = ''
        for word in ngram:
            acr += word[0].upper()
        ngrams_to_acronyms[TOKEN_BREAK_SYMBOL.join(ngram)] = acr

    docs = glob.glob(FILES_TO_PARSE)
    pool = multiprocessing.Pool(10)
    func = partial(extract_train_data_from_report_2, ngrams_to_extract, ngrams_to_acronyms)
    results = pool.map(func, docs)
    pool.close()
    pool.join()

    # OLD GENERATION METHOD
    #with open(outfile, 'w') as out:
    #    for result in results:
    #        for sentence in result:
    #            out.write(sentence[0] + '\t' + sentence[1] + '\n')

    with open(outfile, 'w') as out:
        for result in results:
            for point in result:
                sentence, start_index, expanded, acronym = point
                end_index = start_index + len(acronym)
                out.write('\t'.join([acronym, expanded, str(start_index), str(end_index), sentence]) + '\n')


def extract_train_data_from_report(ngrams_to_extract, ngrams_to_acronyms, report_path):
    note = preprocess.read_note(report_path)
    words = nltk.word_tokenize(note)
    results = []
    for ngram in ngrams_to_extract:
        ngram_start_indices = find_ngram_in_tokens(ngram, words)
        joined_ngram = TOKEN_BREAK_SYMBOL.join(ngram)
        insensitive_acronym_regex = re.compile(re.escape(joined_ngram), re.IGNORECASE)
        for start_index in ngram_start_indices:
            # Extract 20 tokens to either side of the ngram.
            tokens_to_extract = words[max(0, start_index-20):min(len(words), start_index + 20 + len(ngram))]
            no_acr_sentence = TOKEN_BREAK_SYMBOL.join(tokens_to_extract).replace(' .', '.').replace(' ,', ',').replace(' ;', ';')
            acr_sentence = insensitive_acronym_regex.sub(ngrams_to_acronyms[joined_ngram], no_acr_sentence)
            results.append((no_acr_sentence, acr_sentence))
    return results


# Second version of extract_train_data_from_report, which outputs data in
# format more similar to UMN setup.
def extract_train_data_from_report_2(ngrams_to_extract, ngrams_to_acronyms, report_path):
    note = preprocess.read_note(report_path)
    words = nltk.word_tokenize(note)
    results = []

    for ngram in ngrams_to_extract:
        ngram_start_indices = find_ngram_in_tokens(ngram, words)
        joined_ngram = TOKEN_BREAK_SYMBOL.join(ngram)
        acronym = ngrams_to_acronyms[joined_ngram]
        insensitive_acronym_regex = re.compile(re.escape(joined_ngram), re.IGNORECASE)
        for start_index in ngram_start_indices:
            # Extract 40 tokens to either side of the ngram.
            tokens_to_extract = words[max(0, start_index-20):min(len(words), start_index + 20 + len(ngram))]
            no_acr_sentence = TOKEN_BREAK_SYMBOL.join(tokens_to_extract).replace(' .', '.').replace(' ,', ',').replace(' ;', ';')
            acr_sentence = insensitive_acronym_regex.sub(acronym, no_acr_sentence)
            new_start_index = acr_sentence.index(acronym)
            results.append((acr_sentence, new_start_index, joined_ngram, acronym))
    return results

def find_ngram_in_tokens(ngram, tokens):
    ngram_length = len(ngram)
    indices = []
    for ind in range(0, len(tokens) - len(ngram) + 1):
        if ngram == [token.lower() for token in tokens[ind:ind+ngram_length]]:
            indices.append(ind)
    return indices
