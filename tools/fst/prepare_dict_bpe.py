#!/usr/bin/env python3
# encoding: utf-8

import sys

# sys.argv[1]: e2e model unit file(lang_char.txt)
# sys.argv[2]: raw lexicon file
# sys.argv[3]: output lexicon file
# sys.argv[4]: bpemodel

# create table with unit
unit_table = set()
with open(sys.argv[1], 'r', encoding='utf8') as fin:
    for line in fin:
        unit = line.split()[0]
        unit_table.add(unit)

def contain_oov(units):
    for unit in units:
        if unit not in unit_table: # compare character not token
            return True
    return False


bpemode = len(sys.argv) > 4
print(bpemode)
if bpemode:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(sys.argv[4])
    
    lexicon_table = set()
    with open(sys.argv[2], 'r', encoding='utf8') as fin, \
            open(sys.argv[3], 'w', encoding='utf8') as fout:
        for line in fin:
            word = line.split()[0]
            print(word)
            if word in lexicon_table:
                continue
            pieces = sp.EncodeAsPieces(word) # list of tokens
            if contain_oov(pieces):
                print('Ignoring words {}, which contains oov unit'.format(''.join(word).strip('▁')))
                continue
            chars = ' '.join([p if p in unit_table else '<unk>' for p in pieces])    
            fout.write('{} {}\n'.format(word, chars))
            lexicon_table.add(word)
