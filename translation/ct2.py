import re

import ctranslate2

import sentencepiece as spm

from translation.interface import ITranslator


class SugoiCT2Translator(ITranslator):
    def __init__(self, model_dir, cuda=False):
        try:
            self.model = ctranslate2.Translator(model_dir + '/ct2Model', device='cuda' if cuda else 'cpu')
            self.tokenizer = spm.SentencePieceProcessor(model_dir + '/spmModels/spm.ja.nopretok.model')
            self.detokenizer = spm.SentencePieceProcessor(model_dir + '/spmModels/spm.en.nopretok.model')
        except RuntimeError as e:
            raise RuntimeError('Could not load Sugoi Offline Model. Please make sure all files have been placed correctly.')
        except OSError as e:
            raise OSError('Could not find all files of Sugoi Offline Model. Please make sure all files have been placed correctly.')


    def batch(self, sentences):
        tokenized = self.tokenizer.Encode(sentences, out_type=str)
        translated = self.model.translate_batch(tokenized, max_input_length=0, max_decoding_length=1024)
        return [processing(x) for x in self.detokenizer.Decode([x.hypotheses[0] for x in translated])]


def processing(string):
    return re.sub(r'(.)\1{5,}$', r'\1' * 5, string)
