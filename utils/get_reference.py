import os
import re
import shutil
path_to_root = ''

import string
import make_meeting_list

punctuations = string.punctuation

dataset_id = 'icsi'
development_or_test = 'test'
path_to_from = path_to_root + 'resources/data/meeting/' + dataset_id + '/'
path_to_to = path_to_root + 'dataset/utterance/meeting/ami/'

ids = make_meeting_list.ami_reference_list if dataset_id == 'ami' else make_meeting_list.icsi_reference_list

for id in ids:
    if dataset_id == 'ami':
        i = open(path_to_from + id + '.ducref.abstract', 'r')
        d_i = open(path_to_from + id + '.ducref.decisions', 'r')
        a_i = open(path_to_from + id + '.ducref.actions', 'r')
        p_i = open(path_to_from + id + '.ducref.problems', 'r')
        o = open(path_to_to + id + '_reference.txt', 'w')

        content = ''.join(l for l in i.read() if l not in punctuations)
        content = re.sub(' +', ' ', content)
        content = content.lower()
        o.write("minutes: " + content)

        content = ''.join(l for l in d_i.read() if l not in punctuations)
        content = re.sub(' +', ' ', content)
        content = content.lower()
        if content != "" and content != "na\n":
            o.write("decision: " + content)

        content = ''.join(l for l in a_i.read() if l not in punctuations)
        content = re.sub(' +', ' ', content)
        content = content.lower()
        if content != "" and content != "na\n":
            o.write("action: " + content)

        content = ''.join(l for l in p_i.read() if l not in punctuations)
        content = re.sub(' +', ' ', content)
        content = content.lower()
        if content != "" and content != "na\n":
            o.write("problem: " + content)

        print(id, 'reference words count:', len(content.split()))
        # o.write(content)

        i.close()
        a_i.close()
        d_i.close()
        p_i.close()
        o.close()
    elif dataset_id == 'icsi':
        i = open(path_to_from + id + '.ducref.abstract', 'r')
        d_i = open(path_to_from + id + '.ducref.decisions', 'r')
        a_i = open(path_to_from + id + '.ducref.progress', 'r')
        p_i = open(path_to_from + id + '.ducref.problems', 'r')
        o = open(path_to_to + id + '_reference.txt', 'w')

        content = ''.join(l for l in i.read() if l not in punctuations)
        content = re.sub(' +', ' ', content)
        content = content.lower()
        o.write("minutes: " + content)

        content = ''.join(l for l in d_i.read() if l not in punctuations)
        content = re.sub(' +', ' ', content)
        content = content.lower()
        if content != "" and content != "na\n":
            o.write("decision: " + content)

        content = ''.join(l for l in a_i.read() if l not in punctuations)
        content = re.sub(' +', ' ', content)
        content = content.lower()
        if content != "" and content != "na\n":
            o.write("progress: " + content)

        content = ''.join(l for l in p_i.read() if l not in punctuations)
        content = re.sub(' +', ' ', content)
        content = content.lower()
        if content != "" and content != "na\n":
            o.write("problem: " + content)

        print(id, 'reference words count:', len(content.split()))
        # o.write(content)

        i.close()
        a_i.close()
        d_i.close()
        p_i.close()
        o.close()
