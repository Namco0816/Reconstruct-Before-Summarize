import datasets
import re

def remove_extra_whitespace_tabs(example):
    #pattern = r'^\s+$|\s+$'
    def process_text(text):
        text = re.sub(r'\s(\')\s([a-z])', r'\1\2', text) # pattern: ( ' t) -> ('t)
        text = re.sub(r'\s(\')([a-z])', r'\1\2', text) # pattern: ( 's) -> ('s)
        text = re.sub(r'(\w)\s(n\'t)', r'\1\2', text) # pattern: (was n't) -> (wasn't)
        text = re.sub(r'(\wn)\s(na)', r'\1\2', text) # pattern: (gon na) -> (gonna)
        text = re.sub(r'\s(\-)', r'\1', text)
        text = re.sub(r'(\-)\s', r'\1', text)
        text = re.sub(r'\s([,?.!()"])', r'\1', text)
        text = re.sub(r'\s([a-z]+?)(\s[a-z]:)[\W\D]', r' \1\2', text)
        text = re.sub(r'\s([a-z]+?)(\s[a-z]:)', r'\r\n\1\2', text)
        text = re.sub(r'\r\n([a-z]+?)(\s[a-z]:)(\S)', r'\r\n\1\2 \3', text)
        text = re.sub(r'([,?.!()":])([,?.!()":])', r'\1', text)
        text = text.replace("_", '')

        return text
    example['utterances'] = process_text(example['utterances'])
    example['sub_summary'] = process_text(example['sub_summary'])
    return example


ami = datasets.load_from_disk('dataset/icsi_en_new')
ami = ami.map(remove_extra_whitespace_tabs)

ami.save_to_disk('dataset/icsi_en')
