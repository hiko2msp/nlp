#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re
from six.moves import urllib
import zipfile
from janome.tokenizer import Tokenizer


url = 'http://www.aozora.gr.jp/cards/000148/files/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

ruby_match = re.compile(r'《.*》')
brace_match = re.compile(r'［.*］')
stop_words = re.compile(r'[「」\r]')
expranation_match = re.compile(r'-{10,}[^-]*-{10,}')

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0])
        text = data.decode('sjis')
        text = ruby_match.sub('', text)
        text = brace_match.sub('', text)
        text = stop_words.sub('', text)
        text = expranation_match.sub('', text)
    return text

def token_generator(text):
    tokenizer = Tokenizer()
    for text_line in text.split('\n'):
        for token in tokenizer.tokenize(text_line):
            yield token.surface

def read_and_save(filename, output_filename):
    with open(output_filename, 'w') as f:
        f.write(' '.join(
            list(token_generator(read_data(filename)))
        ))


if __name__ == '__main__':
    # 夏目漱石の小説データのダウンロード
    maybe_download('789_ruby_5639.zip', 350404) # 吾輩は猫である
    maybe_download('752_ruby_2438.zip', 94410) # 坊っちゃん
    maybe_download('773_ruby_5968.zip', 153688) # こころ

    read_and_save('789_ruby_5639.zip', 'wagahai.txt')
    read_and_save('752_ruby_2438.zip', 'bochan.txt')
    read_and_save('773_ruby_5968.zip', 'kokoro.txt')
