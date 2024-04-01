import os
import re
import shutil
path_to_root = ''

import string
import make_meeting_list

punctuations = string.punctuation

dataset_id = 'ami_zh'
development_or_test = 'test'
path_to_from = path_to_root + 'resources/data/meeting/' + dataset_id + '/'
path_to_to = path_to_root + 'dataset/utterance/meeting/ami_zh/'

ids = make_meeting_list.ami_reference_list if dataset_id == 'ami_zh' else make_meeting_list.icsi_reference_list

for id in ids:
    if dataset_id == 'ami_zh':
        abstract_i = open(path_to_from + id + '.ducref.abstract.zh', 'r')
        decision_i = open(path_to_from + id + '.ducref.decisions.zh', 'r')
        actions_i = open(path_to_from + id + '.ducref.actions.zh', 'r')
        problems_i = open(path_to_from + id + '.ducref.problems.zh', 'r')

        o = open(path_to_to + id + '_reference.txt', 'w')

        content = "摘要: " + ''.join(l for l in abstract_i.read() if l not in punctuations) 
        content = content + "决策: " + ''.join(l for l in decision_i.read() if l not in punctuations)
        content = content + "行动: " + ''.join(l for l in actions_i.read() if l not in punctuations)
        content = content + "问题: " + ''.join(l for l in problems_i.read() if l not in punctuations)

        content = re.sub(' +', ' ', content)
        print(id, 'reference words count:', len(content.split()))
        o.write(content)

        abstract_i.close()
        decision_i.close()
        actions_i.close()
        problems_i.close()
        o.close()
    elif dataset_id == 'icsi':
        if development_or_test == 'development':
            i = open(path_to_from + id + '.ducref.longabstract', 'r')
            o = open(path_to_to + id + '_reference.txt', 'w')

            content = ''.join(l for l in i.read() if l not in punctuations)
            content = re.sub(' +', ' ', content)
            content = content.lower()
            print(id, 'reference words count:', len(content.split()))
            o.write(content)

            i.close()
            o.close()
        elif development_or_test == 'test':
            for idx, key in enumerate(['beata', 's9553330', 'vkaraisk']):
                i = open(path_to_from + id + '.ducref.' + key + '.longabstract', 'r')
                o = open(path_to_to + id + '_reference' + str(idx) + '.txt', 'w')

                content = ''.join(l for l in i.read() if l not in punctuations)
                content = re.sub(' +', ' ', content)
                content = content.lower()
                print(id, key, 'reference words count:', len(content.split()))
                o.write(content)

                i.close()
                o.close()