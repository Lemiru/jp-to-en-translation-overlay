from enum import Enum


class OCRSelect(Enum):
    MangaOCR = 0
    PaddleOCR = 1


class TranslationSelect(Enum):
    DeepL = 0
    Sugoi = 1


class ParagraphDetectionTypes(Enum):
    NoDetection = 0
    Horizontal = 1
    Vertical = 2


class BinarizationTypes(Enum):
    NoBinarization = 0
    Normal = 1
    Inverted = 2
    NormalWithNegC = 3
    InvertedWithNegC = 4
