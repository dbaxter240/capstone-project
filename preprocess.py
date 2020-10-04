import re
import glob
from tqdm import tqdm
import nltk
import pickle


def split_notes(base_file_loc='C:/Users/Baxter/Desktop/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv', output_loc='C:/Capstone/Reports/'):
    '''
    Method to parse the MIMIC data NOTEEVENTS file and output a series of text files, one containing each report.
    '''
    with open(base_file_loc, 'r') as f:
        f.readline() # Header line

        line = f.readline() # Start of first report
        next_file_name = line.split(',')[0] # Name file after row id

        line = f.readline()
        report_text = ''

        count = 0
        while line:

            count += 1
            if count % 100000 == 0:
                print("Count: " + str(count))
            # Regex for header of new note -- ROW_ID,SUBJECT_ID,HADM_ID,CHART_DATE
            while line and not re.match(r'^[0-9]+,[0-9]+,[0-9]+,[0-9]+-[0-9]+-[0-9]+,', line):
                report_text = report_text + line
                line = f.readline()

            # Processing of Report Text to make them "prettier" and easier to parse.
            report_text = re.sub(r'[_]{3,}', '', report_text)
            #report_text = re.sub(r'\r\n[ \t]*[_]+[ \t]*\r\n', '\r\n\r\n', report_text) # Substitutes out lines in reports which are only underscores -- replaces them with blank lines.

            phi_tag = re.search(r'\[\*\*.*?\*\*\]', report_text)
            while phi_tag:
                txt = report_text[phi_tag.start():phi_tag.end()].lower()
                if 'name' in txt:
                    replacement_text = 'Baxter'
                elif 'hospital' in txt:
                    replacement_text = 'Mayo Clinic'
                elif 'location' in txt:
                    replacement_text = 'Minnesota'
                elif 'phone' in txt:
                    replacement_text = '123-456-7890'
                else:
                    replacement_text = txt[3:len(txt)-3] # Just remove '[**' and '**]'

                report_text = report_text[:phi_tag.start()] + replacement_text + report_text[phi_tag.end():]
                phi_tag = re.search(r'\[\*\*.*?\*\*\]', report_text)

            with open(output_loc + next_file_name + '.txt', 'w') as out:
                out.write(report_text)
            report_text = ''

            if line:
                next_file_name = line.split(',')[0] # Name file after row id
                line = f.readline()


def output_all_umn_acronyms(input='C:/Capstone/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt', output='C:/Capstone/UMNacronyms.txt'):
    '''
    Analysis method used to read all lines from the UMN acronyms dataset, and output a file containing the set of all acronyms/abbreviations
    and all their possible meanings.
    '''
    acronym_meanings = {}
    with open(input, 'r') as inp:
        line = inp.readline()

        while line:
            splitLine = line.split('|')
            if splitLine[0] not in acronym_meanings.keys():
                acronym_meanings[splitLine[0]] = []
            if splitLine[1] not in acronym_meanings[splitLine[0]]:
                acronym_meanings[splitLine[0]].append(splitLine[1])
            line = inp.readline()

    with open(output, 'w+') as out:
        acronyms = list(acronym_meanings.keys())
        acronyms.sort()
        for acronym in acronyms:
            text = acronym + ' : ' + ','.join(acronym_meanings[acronym])
            out.write(text + '\n')

def translate_umn_data(acronyms_to_keep_file='Valid_Acronyms_in_UMN_Data.txt', umn_data_file='AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt', output_file='umn-test-data.txt'):
    '''
    Preprocessing method. Reads a file containing what acronyms/abbreviations from the UMN dataset we want to keep (currently a curated list containing only)
    the acronyms.) Also reads the UMN acronyms/abbreviations dataset, and generates a new file where each line is the sentence from the UMN dataset with the
    acronym expanded and the sentence from the UMN dataset with the acronym unchanged. These two sentences are tab separated. Only sentences with an acronym
    designated as one that should be kept are output.
    '''
    # Read in the valid acronyms file
    valid_acronyms = {}
    with open(acronyms_to_keep_file, 'r') as inp:
        line = inp.readline()
        while line:
            acronym, meanings = line.split(':')
            meanings_list = meanings.split(',')
            valid_acronyms[acronym.strip()] = [x.strip() for x in meanings_list]
            line = inp.readline()

    # Convert data in UMN dataset that contains a "valid acronym" into format
    # <Text w/o acronym> : <Text with acronym>
    output_data = {}
    with open(umn_data_file, 'r') as inp:
        line = inp.readline()
        while line:
            acronym, meaning, _1, begin, end, _2, text = line.split('|')
            begin = int(begin)
            end = int(end)
            text_to_substitute = meaning
            if '/' in text_to_substitute:
                text_to_substitute = text_to_substitute[:text_to_substitute.index('/')]

            text_wo_acronym = ''
            if ':' in meaning:
                split_meaning = meaning.split(':')
                if split_meaning[1] in valid_acronyms.keys() and split_meaning[0] in valid_acronyms[split_meaning[1]]:
                    acronym = split_meaning[1]
                    meaning = split_meaning[0]
                    text_to_substitute = meaning
                    if '/' in text_to_substitute:
                        text_to_substitute = text_to_substitute[:text_to_substitute.index('/')]
                    text_wo_acronym = text[:begin] + text_to_substitute + text[end+1:]
                    text = text[:begin] + acronym + text[end+1:]

            elif acronym in valid_acronyms.keys() and meaning in valid_acronyms[acronym]:
                text_wo_acronym = text[:begin] + text_to_substitute + text[end+1:]

            if text_wo_acronym != '':
                if acronym not in output_data.keys():
                    output_data[acronym] = {}
                output_data[acronym][text_wo_acronym] = text

            line = inp.readline()

    # Now write out the generated data..
    sorted_keys = list(output_data.keys())
    sorted_keys.sort()
    with open(output_file, 'w') as out:
        for acronym in sorted_keys:
            for text_wo_acronym in output_data[acronym].keys():
                line = acronym + '\t' + text_wo_acronym.replace('\t', '').replace('\n', '') + '\t' + output_data[acronym][text_wo_acronym].replace('\t', '').replace('\n', '') + '\n'
                out.write(line)


def find_nbest_bigrams(alltokens, outfile=None, n=10):
    '''
    Given a set of tokens, finds the n best bigrams according to nltk BigramCollocationFinder
    and writes them if the outfile parameter is provided.
    '''
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    min_freq = 5

    print('Generating bigram finder from tokens...')
    finder = nltk.collocations.BigramCollocationFinder.from_words(alltokens)
    print('Applying frequency filter...')
    finder.apply_freq_filter(min_freq)
    print('Finding best bigrams...')
    best_bigrams = finder.nbest(bigram_measures.pmi, n)

    best_bigrams = [bigram for bigram in best_bigrams if not all([len(word) <= 4 for word in bigram])]

    # Write out results
    if outfile:
        out = open(outfile, 'w')
        for bi in best_bigrams:
            # Do not include if every  token in the ngram is length <= 4
            out.write(', '.join(bi) + '\n')
        out.close()

    return best_bigrams


def find_nbest_trigrams(alltokens, outfile=None, n=10):
    '''
    Given a set of tokens, finds the n best trigrams according to nltk BigramCollocationFinder
    and writes them if the outfile parameter is provided.
    '''
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    min_freq = 5

    print('Generating trigram finder from tokens...')
    finder = nltk.collocations.TrigramCollocationFinder.from_words(alltokens)
    print('Applying frequency filter...')
    finder.apply_freq_filter(min_freq)
    print('Finding best trigrams...')
    best_trigrams = finder.nbest(trigram_measures.pmi, n)

    best_trigrams = [trigram for trigram in best_trigrams if not all([len(word) <= 4 for word in trigram])]

    # Write out results
    if outfile:
        out = open(outfile, 'w')
        for tri in best_trigrams:
            # Do not include if every  token in the ngram is length <= 4
            out.write(', '.join(tri) + '\n')
        out.close()

    return best_trigrams


def find_all_best_ngrams(alltokens, outfile=None, n=100):
    '''
    Finds the n best trigrams and the n best bigrams in the tokens provided, and merges them into one list
    If a bigram is either a prefix or a suffix of a trigram found, that bigram is ommitted.
    '''
    trigrams = find_nbest_trigrams(alltokens, None, n)
    bigrams = find_nbest_bigrams(alltokens, None, n)

    # Merging ngrams
    ngrams = [trigram for trigram in trigrams]
    for bigram in bigrams:
        add_bi = True
        for trigram in trigrams:
            if ','.join(trigram).startswith(','.join(bigram)) or ','.join(trigram).endswith(','.join(bigram)):
                add_bi = False
                print('Not adding bigram ' + ','.join(bigram) + ' because it conflicts with trigram ' + ','.join(trigram))
                break;
        if add_bi:
            ngrams.append(bigram)

    if outfile:
        out = open(outfile, 'w')
        for ngram in ngrams:
            # Do not include if every  token in the ngram is length <= 4
            out.write(', '.join(ngram) + '\n')
        out.close()

    return ngrams



def pickle_corpus_tokens(base_loc='C:\\Capstone\\Reports\\', doc_group='*[12345]1.txt', output_loc=None):
    '''
    Parses a set of reports into a series of tokens. The doc_group param specifies (as regex)
    what note ids to parse. If output_loc is provided, the results will be written out as a pkl
    object.
    '''

    alltokens = []
    docs = glob.glob(base_loc + doc_group)
    for doc in tqdm(docs):
        text = read_note(doc)
        text = text.lower()
        text = text.replace('/', ' ')
        text = text.replace('\\', ' ')
        text = text.replace('-', ' ')
        tokens = nltk.word_tokenize(text)
        tokens = ['NONALPHA' if not token.isalpha() else token for token in tokens]
        for token in tokens:
            alltokens.append(token)
    if output_loc:
        out = open(output_loc, 'wb')
        pickle.dump(alltokens, out)
        out.close()
    return alltokens


def generate_ngrams_for_report_groups():
    '''
    This method iterates over the reports, roughly 5% of reports at a time.
    For each group of 5% of reports, it generates the best 3- and 4-grams it
    can find, and writes them to the ngrams directory. This is meant for analysis
    to see if there is much overlap between the ngrams found in each set of reports
    or not.

    Right now, this method is only looking at bi/trigrams, because quadgrams have
    seemed questionable.
    '''
    report_groups = ['*[12345]0.txt', '*[67890]0.txt',
        '*[12345]1.txt', '*[67890]1.txt',
        '*[12345]2.txt', '*[67890]2.txt',
        '*[12345]3.txt', '*[67890]3.txt',
        '*[12345]4.txt', '*[67890]4.txt',
        '*[12345]5.txt', '*[67890]5.txt',
        '*[12345]6.txt', '*[67890]6.txt',
        '*[12345]7.txt', '*[67890]7.txt',
        '*[12345]8.txt', '*[67890]8.txt',
        '*[12345]9.txt', '*[67890]9.txt']

    for report_group in report_groups:
        print('Starting analysis on reports group ' + report_group)
        tokens = pickle_corpus_tokens(doc_group=report_group)
        trigrams = find_all_best_ngrams(tokens, outfile='ngrams/' + report_group[1:], n=500)

    return None

def load_all_tokens(path):
    return pickle.load(open(path, 'rb'))

def read_note(note_id):
    f = open(note_id)
    text = ''.join(f.readlines())
    f.close()
    return text


def test_number_of_groups_method(n):
    report_groups = ['*[12345]0.txt', '*[67890]0.txt',
        '*[12345]1.txt', '*[67890]1.txt',
        '*[12345]2.txt', '*[67890]2.txt',
        '*[12345]3.txt', '*[67890]3.txt',
        '*[12345]4.txt', '*[67890]4.txt',
        '*[12345]5.txt', '*[67890]5.txt',
        '*[12345]6.txt', '*[67890]6.txt',
        '*[12345]7.txt', '*[67890]7.txt',
        '*[12345]8.txt', '*[67890]8.txt',
        '*[12345]9.txt', '*[67890]9.txt']

    all_ngrams = [load_ngrams_from_file('ngrams/' + group[1:]) for group in report_groups]
    ngrams_to_keep = set()
    for ngram_list in all_ngrams:
        for ngram in ngram_list:
            count = 0
            for ngram_list2 in all_ngrams:
                if ngram in ngram_list2:
                    count += 1
                    if count >= n:
                        ngrams_to_keep.add(','.join(ngram))
    return ngrams_to_keep

def load_ngrams_from_file(input_file):
    inp = open(input_file, 'r')
    lines = inp.readlines()
    inp.close()
    return [line.replace(' ', '').replace('\n', '').split(',') for line in lines]
