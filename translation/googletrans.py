import asyncio

import googletrans

from translation.interface import ITranslator


class GoogleTranslateTranslator(ITranslator):
    def __init__(self):
        self.translator = googletrans.Translator()

    def batch(self, sentences):
        self.translator = googletrans.Translator()
        translated = asyncio.run(self._get_translations(sentences))
        return [result.text for result in translated]

    async def _get_translations(self, sentences):
        translations = await self.translator.translate(sentences, src='ja', dest='en')
        return translations
