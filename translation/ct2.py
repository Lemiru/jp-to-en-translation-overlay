import re

import ctranslate2

import sentencepiece as spm

from translation.interface import ITranslator


class SugoiCT2Translator(ITranslator):
    def __init__(self, model_dir, cuda=False):
        self.model = ctranslate2.Translator(model_dir + '/ct2Model', device='cuda' if cuda else 'cpu')
        self.tokenizer = spm.SentencePieceProcessor(model_dir + '/spmModels/spm.ja.nopretok.model')
        self.detokenizer = spm.SentencePieceProcessor(model_dir + '/spmModels/spm.en.nopretok.model')

    def batch(self, sentences):
        tokenized = self.tokenizer.Encode(sentences, out_type=str)
        translated = self.model.translate_batch(tokenized, max_input_length=0, max_decoding_length=1024)
        return [processing(x) for x in self.detokenizer.Decode([x.hypotheses[0] for x in translated])]


def processing(string):
    return re.sub(r'(.)\1{5,}$', r'\1' * 5, string)
