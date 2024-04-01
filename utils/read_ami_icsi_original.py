import codecs
import math
import re
import operator
from nltk import PerceptronTagger
import nltk
import pandas as pd
import string
import make_meeting_list
import glob
import shutil

role_dict = {
    'PM': "Project Manager",
    'ID': "Industrial Designer",
    'UI': "User Interface Designer",
    'ME': 'Marketing Expert',
}

def load_stopwords(path):
    stopwords = set([])

    for line in codecs.open(path, 'r', 'utf-8'):
        if not re.search('^#', line) and len(line.strip()) > 0:
            stopwords.add(line.strip().lower())  # lowercase
    print(stopwords)

    return stopwords


def load_filler_words(path):
    with open(path, 'r+') as f:
        filler = f.read().splitlines()

    return filler


def clean_utterance(utterance, filler_words):
    utt = utterance
    # replace consecutive unigrams with a single instance
    utt = re.sub('\\b(\\w+)\\s+\\1\\b', '\\1', utt)
    # same for bigrams
    utt = re.sub('(\\b.+?\\b)\\1\\b', '\\1', utt)
    # strip extra white space
    utt = re.sub(' +', ' ', utt)
    # strip leading and trailing white space
    utt = utt.strip()

    # remove filler words # highly time-consuming
    utt = ' ' + utt + ' '
    for filler_word in filler_words:
        utt = re.sub(' ' + filler_word + ' ', ' ', utt)
        utt = re.sub(' ' + filler_word.capitalize() + ' ', ' ', utt)

    return utt


def clean_text(text, stopwords, remove_stopwords=True, pos_filtering=False, stemming=True, lower_case=False):
    if lower_case:
        # convert to lower case
        text = text.lower()
    # strip extra white space
    text = re.sub(' +', ' ', text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    tokens = text.split(' ')

    # remove punctuation
    tokens = [t for t in tokens if t not in string.punctuation]

    if pos_filtering:
        tagger = PerceptronTagger()
        # apply POS-tagging
        tagged_tokens = tagger.tag(tokens)
        # retain only nouns and adjectives
        tokens = [item[0] for item in tagged_tokens if item[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]

    if remove_stopwords:
        # remove stopwords
        tokens = [token for token in tokens if token.lower() not in stopwords]

    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed

    return (tokens)


def read_ami_icsi(path, filler_words):
    asr_output = pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=['ID', 'start', 'end', 'letter', 'role', 'A', 'B', 'C', 'utt']
    )

    utterances = []
    for tmp in zip(asr_output['role'].tolist(), asr_output['utt'].tolist()):
        role, utt = tmp
        if type(utt) is float:
            if math.isnan(utt):
                continue
        for ch in ['{vocalsound}', '{gap}', '{disfmarker}', '{comment}', '{pause}', '@reject@']:
            # utt = utt.replace(ch, "")
            utt = re.sub(ch, '', utt)

        utt = re.sub("'Kay", 'Okay', utt)
        utt = re.sub("'kay", 'Okay', utt)
        utt = re.sub('"Okay"', 'Okay', utt)
        utt = re.sub("'cause", 'cause', utt)
        utt = re.sub("'Cause", 'cause', utt)
        utt = re.sub('"cause"', 'cause', utt)
        utt = re.sub('"\'em"', 'them', utt)
        utt = re.sub('"\'til"', 'until', utt)
        utt = re.sub('"\'s"', 's', utt)

        # l. c. d. -> lcd
        # t. v. -> tv
        utt = re.sub('h. t. m. l.', 'html', utt)
        utt = re.sub(r"(\w)\. (\w)\. (\w)\.", r"\1\2\3", utt)
        utt = re.sub(r"(\w)\. (\w)\.", r"\1\2", utt)
        utt = re.sub(r"(\w)\.", r"\1", utt)

        # clean_utterance, remove filler_words
        utt = clean_utterance(utt, filler_words=filler_words)

        # strip extra white space
        utt = re.sub(' +', ' ', utt)
        # strip leading and trailing white space
        utt = utt.strip()

        if utt != '' and utt != '.' and utt != ' ':
            utterances.append((role, utt))

    # remove duplicate utterances per speaker
    utterances = sorted(set(utterances), key=utterances.index)

    utterances_indexed = zip(range(len(utterances)), list(zip(*utterances))[0], list(zip(*utterances))[1])

    return utterances_indexed


def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total

path_to_root = ''
domain     = 'meeting' # meeting
dataset_id = 'icsi'     # ami, icsi
language   = 'en'      # en
source     = 'manual'     # asr, manual

# #########################
# ### RESOURCES LOADING ###
# #########################
if domain == 'meeting':
    path_to_stopwords    = path_to_root + 'resources/stopwords/meeting/stopwords.' + language + '.dat'
    path_to_filler_words = path_to_root + 'resources/stopwords/meeting/filler_words.' + language + '.txt'
    stopwords = load_stopwords(path_to_stopwords)
    filler_words = load_filler_words(path_to_filler_words)

    if dataset_id == 'ami':
        ids = make_meeting_list.ami_meeting_list
    elif dataset_id == 'icsi':
        ids = make_meeting_list.icsi_meeting_list
    elif dataset_id == 'ami_zh':
        ids = make_meeting_list.ami_zh_meeting_list
    elif dataset_id == 'ami_with_action':
        ids = []
        raw_ids = make_meeting_list.ami_zh_meeting_list
        for id in raw_ids:
            sub_ids = glob.glob(f'resources/data/meeting/ami_with_action/{id}_*.da')
            sub_ids = [i.split("/")[-1].strip(".da") for i in sub_ids]
            ids += sub_ids

corpus = {}
for id in ids:
    if source == 'asr':
        path = path_to_root + 'resources/data/meeting/' + dataset_id + '/' + id + '.da-asr'
    elif source == 'manual':
        path = path_to_root + 'resources/data/meeting/' + dataset_id + '/' + id + '.da'

    # filler words will be removed during corpus loading
    print(id)
    corpus[id] = read_ami_icsi(path, filler_words)

min_words = 2

for id in ids:

    utterances_indexed = corpus[id]

    # #####################################
    # ### Pre-processing for Clustering ###
    # #####################################
    utterances_processed = []
    lists_of_terms = []
    utterances_remain = []
    current_role = ''
    for utterance_indexed in utterances_indexed:
        index, role, utt = utterance_indexed
        utt_cleaned = clean_text(
            utt,
            stopwords=stopwords,
            remove_stopwords=False,
            pos_filtering=False,
            stemming=True,
            # clustering based on lowercase form.
            lower_case=True
        )
        
        # remove utterances with less than min_words number of non-stopwords
        if len(utt_cleaned) >= min_words:
            if role != current_role:
                if dataset_id == "ami":
                    utt = role_dict[role] + ": " + utt
                else:
                    utt = role + ": " + utt
                current_role = role
            utterances_processed.append((index, role, ' '.join(utt_cleaned)))
            lists_of_terms.append(utt_cleaned)
            utterances_remain.append(utt)
        else:
           pass
    if utterances_remain == []: 
        print(id)
        continue
    path_to_utterance = path_to_root + 'dataset/utterance/meeting/' + dataset_id + '/'
    with open(path_to_utterance + id + '_utterances.txt', 'w+') as txtfile:
        txtfile.write('\n'.join(utterances_remain))
    # shutil.copyfile(
    #     f'resources/data/meeting/{dataset_id}/{id}.sub_action.txt', 
    #     f'dataset/utterance/meeting/{dataset_id}/{id}.sub_action.txt'
    # )