#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import zipfile
from janome.tokenizer import Tokenizer

ruby_match = re.compile(r'《.*》')
brace_match = re.compile(r'［.*］')
stop_words = re.compile(r'[「。、」\r]')

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0])
        text = data.decode('sjis')
        text = ruby_match.sub('', text)
        text = brace_match.sub('', text)
        text = stop_words.sub('', text)
    return text

def token_generator(text):
    tokenizer = Tokenizer()
    for text_line in text.split('\n'):
        for token in tokenizer.tokenize(text_line):
            if token.part_of_speech.split(',')[0] == '名詞':
                yield token.surface

if __name__ == '__main__':
    text = read_data('789_ruby_5639.zip')
    print(list(token_generator(text)))
