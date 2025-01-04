import deepl

from translation.interface import ITranslator


class DeeplTranslator(ITranslator):
    def __init__(self, auth_key):
        self.translator = deepl.Translator(auth_key)

    def batch(self, sentences):
        return [result.text for result in self.translator.translate_text(sentences, target_lang='EN-US')]
