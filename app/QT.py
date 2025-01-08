import time
import configparser
import os
from sys import platform
from typing import Union

from pynput import keyboard

if platform == "win32":
    from ctypes import windll

import cv2
import mss
import numpy as np
import re

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton, QMainWindow, QFrame, QGridLayout, QCheckBox, QWidget,
                               QVBoxLayout, QSlider, QTabWidget, QComboBox, QLineEdit)

from text_detection.CRAFT import CRAFTModel, prepare_for_text_detection
from ocr.manga_ocr import MangaOcrBatchProcessing
from ocr.paddle_ocr import PaddleOcrBatchProcessing
from imageProcessing.processing import check_for_image_difference, create_new_image, draw_boxes, draw_text, process_for_ocr
from translation.interface import ITranslator
from translation.ct2 import SugoiCT2Translator
from translation.deepl import DeeplTranslator
from translation.googletrans import GoogleTranslateTranslator

from common import OCRSelect, ParagraphDetectionTypes, BinarizationTypes


PATH = ''
CONFIG_PATH = ''
CONFIG = configparser.ConfigParser()
DEBUG = False


def create_default_config():
    CONFIG['general'] = {
        'OCR method': '0',
        'translation method': 'Google',
        'font size': '12.0',
        'deepl api key': ''
    }
    CONFIG['preprocessing'] = {
        'preprocess': 'True',
        'resize': '1',
        'histogram equalization': 'False',
        'gaussian blur': '1',
        'binarization type': '0',
        'binarization blocks': '3',
        'binarization c value': '5',
        'post binarization gaussian blur': '1'
    }
    CONFIG['text_detection'] = {
        'paragraph detection': '0',
        'text threshold': '0.8',
        'link threshold': '0.5',
        'low text': '0.4'
    }


def save_config():
    if CONFIG_PATH != '':
        with open(CONFIG_PATH, 'w') as configfile:
            CONFIG.write(configfile)
    else:
        raise ValueError('Config file path has not been set')


class Master(QObject):
    tl_setup = Signal(bool)
    tl_start = Signal()
    tl_tick = Signal(np.ndarray, dict)
    delete_event = Signal()

    def __init__(self, thread):
        super().__init__()
        self.thread = thread
        self.delete_event.connect(self.on_delete)

    def on_delete(self):
        self.thread.exit()


class TranslationWorker(QObject):
    setup_completed = Signal(bool)
    tl_completed = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.craftModel: Union[CRAFTModel, None] = None
        self.mangaOCR: Union[MangaOcrBatchProcessing, None] = None
        self.paddleOCR: Union[PaddleOcrBatchProcessing, None] = None
        self.translator: Union[ITranslator, None] = None
        self.loadedOCR: Union[OCRSelect, None] = None
        self.loadedTranslator: Union[str, None] = None
        self.cuda = False
        self.lastTranslation = None

    def setup(self, cuda=False):
        self.cuda = cuda
        print('cuda set to {}'.format(self.cuda))
        print('starting setup')
        self.craftModel = CRAFTModel(refiner=True, cuda=self.cuda)
        match OCRSelect(int(CONFIG['general']['OCR method'])):
            case OCRSelect.MangaOCR:
                self.mangaOCR = MangaOcrBatchProcessing(cuda=self.cuda)
                self.loadedOCR = OCRSelect.MangaOCR

            case OCRSelect.PaddleOCR:
                self.paddleOCR = PaddleOcrBatchProcessing()
                self.loadedOCR = OCRSelect.PaddleOCR

            case _:
                raise ValueError('invalid OCR method')
        available = SetupWindow.get_available_translations()
        if CONFIG['general']['translation method'] in available:
            match CONFIG['general']['translation method']:
                case 'DeepL':
                    self.translator = DeeplTranslator(CONFIG['general']['DeepL API key'])
                    self.loadedTranslator = 'DeepL'

                case 'Sugoi':
                    self.translator = SugoiCT2Translator('models/sugoi_v4model_ctranslate2', cuda=self.cuda)
                    self.loadedTranslator = 'Sugoi'

                case 'Google':
                    self.translator = GoogleTranslateTranslator()
                    self.loadedTranslator = 'Google'

                case _:
                    raise ValueError('invalid translation method')
            self.setup_completed.emit(False)
        else:
            print('Translation method became unavailable, falling back to Google Translator')
            self.translator = GoogleTranslateTranslator()
            self.loadedTranslator = 'Google'
            CONFIG['general']['translation method'] = 'Google'
            self.setup_completed.emit(True)
        print('Setup Completed')


    def verify_and_load_models(self):
        OCR = OCRSelect(int(CONFIG['general']['OCR method']))
        translator = CONFIG['general']['translation method']
        if self.loadedOCR != OCR:
            match OCR:
                case OCRSelect.MangaOCR:
                    self.mangaOCR = MangaOcrBatchProcessing(cuda=self.cuda)
                    self.loadedOCR = OCRSelect.MangaOCR
                case OCRSelect.PaddleOCR:
                    self.paddleOCR = PaddleOcrBatchProcessing()
                    self.loadedOCR = OCRSelect.PaddleOCR
                case _:
                    raise ValueError('invalid OCR method')

        if self.loadedTranslator != translator:
            self.lastTranslation = None
            match translator:
                case 'DeepL':
                    self.translator = DeeplTranslator(CONFIG['general']['DeepL API key'])
                    self.loadedTranslator = 'DeepL'
                case 'Sugoi':
                    self.translator = SugoiCT2Translator('models/sugoi_v4model_ctranslate2', cuda=self.cuda)
                    self.loadedTranslator = 'Sugoi'
                case 'Google':
                    self.translator = GoogleTranslateTranslator()
                    self.loadedTranslator = 'Google'
                case _:
                    raise ValueError('invalid translation method')

    def translate(self, array: np.ndarray, settings: dict):
        t1 = time.time()
        if settings['preprocess']:
            processed = process_for_ocr(array.copy(), settings['blocks'], settings['c'], settings['resize'], binarization=settings['binarization'],
                                        gaussian_kernel=settings['blur_kernel_size'], gaussian_kernel_post=settings['binarization_blur_kernel_size'],
                                        equalize=settings['equalize'])
            if DEBUG:
                cv2.imwrite('processed.jpg', processed)
        else:
            processed = array.copy()
            settings['resize'] = 1
        boxes = self.craftModel.detect(processed, settings['text_threshold'], settings['link_threshold'], settings['low_text'])
        image_size = (int(array.shape[0] * settings['resize']), int(array.shape[1] * settings['resize']))
        if DEBUG:
            debug_image = array.copy()
            for x in boxes:
                cv2.polylines(debug_image, [np.array(x/settings['resize'], dtype=int)], True, (0, 255, 0), 1)
            cv2.imwrite('debug_image.png', debug_image)
        boxes_for_detection = prepare_for_text_detection(boxes, image_size, paragraph=settings['paragraph_detection'])
        if len(boxes_for_detection) == 0:
            new_image = create_new_image((array.shape[0], array.shape[1], 4))
            self.tl_completed.emit(new_image)
            return
        for_ocr = [processed[x[0][0][1]:x[0][1][1], x[0][0][0]:x[0][1][0]] for x in boxes_for_detection]
        match settings['OCR']:
            case OCRSelect.MangaOCR:
                if len(for_ocr) == 1:
                    texts = [self.mangaOCR.single(for_ocr[0])]
                else:
                    texts = self.mangaOCR.batch(for_ocr)
            case OCRSelect.PaddleOCR:
                texts = self.paddleOCR.batch(for_ocr)
            case _:
                raise ValueError('Incorrect OCR method')

        if DEBUG:
            print(texts)

        japanese_characters_pattern = re.compile(r'[\u3040-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F]')
        filtered_boxes = []
        filtered_texts = []
        for x in range(len(texts)):
            if len(japanese_characters_pattern.findall(texts[x])) >= settings['min_japanese_characters']:
                box = np.array(np.array(boxes_for_detection[x][0]) / settings['resize'], dtype=np.int32)
                filtered_boxes.append([box, [np.array(np.array(box) / settings['resize'], dtype=np.int32) for box in boxes_for_detection[x][1]] if boxes_for_detection[x][1] is not None else None])
                filtered_texts.append(texts[x])
        if DEBUG:
            debug_image = array.copy()
            for x in filtered_boxes:
                cv2.rectangle(debug_image, x[0][0], x[0][1], (255, 0, 0), 2)
                cv2.imwrite('debug_filtered.png', debug_image)
        if self.lastTranslation is None or filtered_texts != self.lastTranslation[0]:
            new_image = draw_boxes(array, filtered_boxes, 9)
            translated = self.translator.batch(filtered_texts)
            if DEBUG:
                print(filtered_texts)
                print(translated)
            final_image = draw_text(new_image, filtered_boxes, translated, max_font_size=settings['font_size'])
            self.lastTranslation = (filtered_texts, translated, final_image)
        else:
            if DEBUG:
                print(self.lastTranslation[0])
                print('No change in detected text')
            final_image = self.lastTranslation[2]
        print('Total Translation Time: {}'.format(time.time()-t1))
        self.tl_completed.emit(final_image)


class Move(QFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold = False
        self.setStyleSheet('background-color: rgb(150, 0, 0)')

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.hold = True

    def mouseMoveEvent(self, event):
        if self.hold:
            self.parent().move(int(event.globalPosition().x()) - 11, int(event.globalPosition().y()) - 11)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.hold = False


class Resize(QFrame):
    def __init__(self, *args, **kwargs):
        super(Resize, self).__init__(*args, **kwargs)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: rgb(0, 0, 150)')
        self.hold = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.hold = True

    def mouseMoveEvent(self, event):
        if self.hold:
            window_pos = self.parent().pos()
            new_w = int(event.globalPosition().x()) + 11 - window_pos.x()
            new_h = int(event.globalPosition().y()) + 11 - window_pos.y()
            if new_w > 50 and new_h > 50:
                self.parent().resize(new_w, new_h)
                self.move(new_w - 22, new_h - 22)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.hold = False


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Preview')
        self.qimg = None
        self.image = QLabel(self)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.image)
        self.setLayout(self.vbox)

    def show_processed_image(self, image: np.ndarray):
        settings = {'OCR': OCRSelect(int(CONFIG['general']['OCR method'])),
                    'translator': CONFIG['general']['translation method'],
                    'preprocess': CONFIG['preprocessing']['preprocess'] == 'True',
                    'resize': float(CONFIG['preprocessing']['resize']),
                    'blur_kernel_size': int(CONFIG['preprocessing']['gaussian blur']),
                    'equalize': CONFIG['preprocessing']['histogram equalization'] == 'True',
                    'binarization': BinarizationTypes(int(CONFIG['preprocessing']['binarization type'])),
                    'blocks': int(CONFIG['preprocessing']['binarization blocks']),
                    'c': int(CONFIG['preprocessing']['binarization c value']),
                    'binarization_blur_kernel_size': int(CONFIG['preprocessing']['post binarization gaussian blur']),
                    'paragraph_detection': ParagraphDetectionTypes(
                        int(CONFIG['text_detection']['paragraph detection'])),
                    'text_threshold': float(CONFIG['text_detection']['text threshold']),
                    'link_threshold': float(CONFIG['text_detection']['link threshold']),
                    'low_text': float(CONFIG['text_detection']['low text']),
                    'min_japanese_characters': 1,
                    'font_size': float(CONFIG['general']['font size'])
                    }
        if settings['preprocess']:
            self.update_preview(process_for_ocr(image, settings['blocks'], settings['c'], settings['resize'], binarization=settings['binarization'],
                                        gaussian_kernel=settings['blur_kernel_size'], gaussian_kernel_post=settings['binarization_blur_kernel_size'],
                                        equalize=settings['equalize']))
        else:
            self.update_preview(image)

    def update_preview(self, image: np.ndarray):
        if DEBUG:
            cv2.imwrite('preview.png', image)
        print(image.shape)
        self.image.resize(image.shape[1], image.shape[0])
        bytes_count = image.data.nbytes
        bytes_per_line = int(bytes_count / image.shape[0])
        if len(image.shape) == 2:
            self.qimg = QImage(image.data, image.shape[1], image.shape[0], bytes_per_line, QImage.Format.Format_Grayscale8)
        elif image.shape[2] == 3:
            self.qimg = QImage(image.data, image.shape[1], image.shape[0], bytes_per_line, QImage.Format.Format_RGB888)
        elif image.shape[2] == 4:
            self.qimg = QImage(image.data, image.shape[1], image.shape[0], bytes_per_line, QImage.Format.Format_RGBA8888_Premultiplied)
        self.image.setPixmap(QPixmap.fromImage(self.qimg))
        self.show()


class OverlayWindow(QMainWindow):
    def __init__(self, thread_master, cuda=False):
        super().__init__()
        self.qimg = None
        self.tlInProgress = False
        self.tlRunning = False
        self.threadMaster = thread_master
        self.threadMaster.tl_setup.emit(cuda)
        self.mss = mss.mss()
        self.timer = QTimer(self)
        self.windowFlags = Qt.SubWindow | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.begin_translation)
        self.timer.start()
        self.resize(800, 600)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(self.windowFlags)
        self.frame = QFrame(self)
        self.frame.setStyleSheet('border: 2px solid rgb(0, 150, 0); border-radius: 4px')
        self.frame.setGeometry(0, 0, 800, 600)
        self.moveWidget = Move(parent=self)
        self.moveWidget.setGeometry(2, 2, 20, 20)
        self.resizeWidget = Resize(parent=self)
        self.resizeWidget.setGeometry(self.size().width() - 2 - 20, self.size().height() - 2 - 20, 20, 20)
        self.image = QLabel(self)
        self.image.setGeometry(2, 2, 800 - (2 * 2), 600 - (2 * 2))
        self.image.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.image.setVisible(False)
        self.settings = {}
        self.lastScreenshot = None

    def resizeEvent(self, event):
        self.frame.resize(event.size().width(), event.size().height())
        self.image.resize(event.size().width() - (2 * 2), event.size().height() - (2 * 2))

    def closeEvent(self, event):
        print('closing')
        self.threadMaster.delete_event.emit()

    def take_screenshot(self):
        x, y = self.pos().x(), self.pos().y()
        w, h = self.size().width(), self.size().height()
        if platform == "win32":
            windll.user32.SetWindowDisplayAffinity(self.winId(), 0x11)
            screenshot = self.mss.grab({'top': y + 2, 'left': x + 2, 'width': w - (2 * 2), 'height': h - (2 * 2)})
            windll.user32.SetWindowDisplayAffinity(self.winId(), 0x0)
        else:
            self.hide()
            screenshot = self.mss.grab({'top': y + 2, 'left': x + 2, 'width': w - (2 * 2), 'height': h - (2 * 2)})
            self.show()
        return screenshot

    def start_overlay(self, settings):
        self.settings = settings
        if DEBUG:
            print(self.settings)
        self.resizeWidget.setVisible(False)
        self.moveWidget.setVisible(False)
        self.image.setVisible(True)
        self.setWindowFlags(self.windowFlags | Qt.WindowTransparentForInput)
        self.show()
        self.threadMaster.tl_start.emit()
        self.tlRunning = True

    def stop_overlay(self):
        self.tlRunning = False
        self.resizeWidget.setVisible(True)
        self.moveWidget.setVisible(True)
        self.image.setVisible(False)
        self.image.clear()
        self.setWindowFlags(self.windowFlags & ~Qt.WindowTransparentForInput)
        self.show()
        self.lastScreenshot = None

    def begin_translation(self):
        if self.tlInProgress or not self.tlRunning:
            pass
        else:
            screenshot = np.array(self.take_screenshot())
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
            if self.lastScreenshot is None or check_for_image_difference(screenshot, self.lastScreenshot):
                self.tlInProgress = True
                self.lastScreenshot = screenshot
                self.threadMaster.tl_tick.emit(screenshot, self.settings)

    def update_overlay(self, image: np.ndarray):
        self.qimg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGBA8888_Premultiplied)
        if DEBUG:
            self.qimg.save('test_qt.png')
        self.image.setPixmap(QPixmap.fromImage(self.qimg))
        self.tlInProgress = False


class QInputWithLabel(QWidget):
    input_changed = Signal()

    def __init__(self, name: str, section: str, input_class: type(QWidget), min_val: Union[int, float] = 0,
                 max_val: Union[int, float] = 100, step: Union[int, float] = 1, combobox_list: Union[list, None] = None,
                 combobox_return_text: bool = False, tooltip: Union[str, None] = None):
        super().__init__()
        self.section = section
        self.name = name
        if self.section not in CONFIG.sections() and self.section != 'DEFAULT':
            CONFIG[self.section] = {}
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.input = input_class()
        if input_class == QSlider:
            self.className = 'QSlider'
            slider_position = float(CONFIG[self.section][self.name])
            if slider_position.is_integer():
                slider_position = int(slider_position)
            self.label = QLabel(f"{self.name} = {slider_position}")
            self.input.setOrientation(Qt.Orientation.Horizontal)
            if step < 1:
                inverted_step = 1/step
                if inverted_step.is_integer():
                    step = int(inverted_step)
                    min_val = int(min_val * step)
                    max_val = int(max_val * step)
                    slider_position = int(slider_position * step)
                    self.input.valueChanged.connect(
                        lambda x: (
                            (
                                self.label.setText(f"{self.name} = {x/step}"),
                                self.update_config(self.section, self.name, x/step)
                            )
                        ))
                else:
                    raise ValueError("Incorrect Step Value")
            else:
                self.input.valueChanged.connect(
                    lambda x: (
                        (
                            self.label.setText(f"{self.name} = {x}"),
                            self.update_config(self.section, self.name, x),
                            self.input_changed.emit()
                        ) if min_val % step == x % step else
                        (
                            self.input.setValue(x + (min_val % step) - (x % step)),
                            self.label.setText(f"{self.name} = {(x + (min_val % step) - (x % step))}"),
                            self.update_config(self.section, self.name, (x + (min_val % step) - (x % step))),
                            self.input_changed.emit()
                        )
                    ))
            self.input.setMinimum(min_val)
            self.input.setMaximum(max_val)
            self.input.setValue(slider_position)
            self.input.setTickInterval(step)
            self.input.setSingleStep(step)
        elif input_class == QComboBox:
            self.className = 'QComboBox'
            if combobox_list is None:
                raise ValueError("combobox_list is required for ComboBox")
            self.label = QLabel(self.name)
            for element in combobox_list:
                self.input.addItem(element)
            if combobox_return_text:
                self.input.setCurrentText(CONFIG[self.section][self.name])
                self.input.currentTextChanged.connect(
                    lambda x: (
                        self.update_config(self.section, self.name, x),
                        self.input_changed.emit()
                    ))
            else:
                self.input.setCurrentIndex(int(CONFIG[self.section][self.name]))
                self.input.currentIndexChanged.connect(
                    lambda x: (
                        self.update_config(self.section, self.name, x),
                        self.input_changed.emit()
                    ))
        elif input_class == QCheckBox:
            self.className = 'QCheckBox'
            self.input.setText(self.name)
            self.input.setChecked((CONFIG[self.section][self.name] == 'True'))
            self.input.stateChanged.connect(
                lambda: (
                    self.update_config(self.section, self.name, self.input.isChecked()),
                    self.input_changed.emit()
                ))
        elif input_class == QLineEdit:
            self.className = 'QLineEdit'
            self.label = QLabel(self.name)
            self.input.textEdited.connect(
                lambda: (
                    self.update_config(self.section, self.name, self.input.text()),
                    self.input_changed.emit()
                ))

        else:
            raise ValueError("Incorrect QWidget type")
        if tooltip is not None:
            self.input.setToolTip(tooltip)
        if input_class != QCheckBox:
            self.layout.addWidget(self.label)
        self.layout.addWidget(self.input)
        self.setLayout(self.layout)

    def modify_input(self, value):
        match self.className:
            case 'QComboBox':
                current = CONFIG[self.section][self.name]
                self.input.clear()
                for element in value:
                    self.input.addItem(element)
                if current in value:
                    self.input.setCurrentText(current)

    def set_value(self, value):
        match self.className:
            case 'QComboBox':
                self.input.setCurrentText(value)

    @staticmethod
    def update_config(section, name, value):
        CONFIG[section][name] = str(value)


class SetupWindow(QMainWindow):
    def __init__(self, overlay: OverlayWindow):
        super().__init__()
        self.setupCompleted = False
        self.setWindowTitle('Translation Overlay')
        self.setFixedSize(325, 425)
        self.startButton = QPushButton('Loading...', parent=self)
        self.startButton.setEnabled(False)
        self.overlay = overlay
        self.preview = PreviewWindow()
        self.overlayStarted = False
        self.startButton.clicked.connect(self.toggle_overlay)
        self.container = QWidget(self)
        self.optionsTabs = QTabWidget(self)

        self.generalOptions = QWidget(self)
        self.ocrComboBox = QInputWithLabel('OCR method', 'general', QComboBox, combobox_list=[x.name for x in OCRSelect])
        self.translatorComboBox = QInputWithLabel('Translation method', 'general', QComboBox, combobox_list=self.get_available_translations(), combobox_return_text=True)
        self.deepLKeyLineEdit = QInputWithLabel('DeepL API Key', 'general', QLineEdit)
        self.fontSizeComboBox = QInputWithLabel('Font size', 'general', QComboBox, combobox_list=[str(10.0 + x/2) for x in range(11)], combobox_return_text=True)
        self.deepLKeyLineEdit.input_changed.connect(lambda: self.translatorComboBox.modify_input(self.get_available_translations()))

        self.generalOptionsLayout = QVBoxLayout(self)
        self.generalOptionsLayout.addWidget(self.ocrComboBox)
        self.generalOptionsLayout.addWidget(self.translatorComboBox)
        self.generalOptionsLayout.addWidget(self.deepLKeyLineEdit)
        self.generalOptionsLayout.addWidget(self.fontSizeComboBox)

        self.generalOptionsLayout.addStretch()
        self.generalOptions.setLayout(self.generalOptionsLayout)
        self.optionsTabs.addTab(self.generalOptions, 'General')

        self.preprocessingOptions = QWidget(self)
        self.preprocessingCheckmark = QInputWithLabel('Preprocess', 'preprocessing', QCheckBox, tooltip="When unchecked other settings in this tab will have no effect")
        self.resizeSlider = QInputWithLabel('Resize', 'preprocessing', QSlider, min_val=1, max_val=4)
        self.equalizationCheckmark = QInputWithLabel('Histogram Equalization', 'preprocessing', QCheckBox, tooltip="Helpful when there's low contrast between text and background.\nNot recommended for high contrast images (for example, black and white manga panels)")
        self.gaussianSlider = QInputWithLabel('Gaussian Blur', 'preprocessing', QSlider, min_val=1, max_val=15, step=2, tooltip="Value of 1 disables blurring")
        self.binarizationComboBox = QInputWithLabel('Binarization type', 'preprocessing', QComboBox, combobox_list=[x.name for x in BinarizationTypes])
        self.cSlider = QInputWithLabel('Binarization C value', 'preprocessing', QSlider, min_val=1, max_val=100)
        self.blockSlider = QInputWithLabel('Binarization Blocks', 'preprocessing', QSlider, min_val=3, max_val=75, step=2)
        self.binarizationGaussianSlider = QInputWithLabel('Post Binarization Gaussian Blur', 'preprocessing', QSlider, min_val=1, max_val=15, step=2, tooltip="Value of 1 disables blurring")
        self.previewButton = QPushButton('Preview')
        self.previewButton.clicked.connect(lambda: self.preview.show_processed_image(np.array(self.overlay.take_screenshot())))

        self.preprocessingOptionsLayout = QVBoxLayout(self)
        self.preprocessingOptionsLayout.addWidget(self.preprocessingCheckmark)
        self.preprocessingOptionsLayout.addWidget(self.resizeSlider)
        self.preprocessingOptionsLayout.addWidget(self.equalizationCheckmark)
        self.preprocessingOptionsLayout.addWidget(self.gaussianSlider)
        self.preprocessingOptionsLayout.addWidget(self.binarizationComboBox)
        self.preprocessingOptionsLayout.addWidget(self.cSlider)
        self.preprocessingOptionsLayout.addWidget(self.blockSlider)
        self.preprocessingOptionsLayout.addWidget(self.binarizationGaussianSlider)
        self.preprocessingOptionsLayout.addWidget(self.previewButton)

        self.preprocessingOptionsLayout.addStretch()
        self.preprocessingOptions.setLayout(self.preprocessingOptionsLayout)
        self.optionsTabs.addTab(self.preprocessingOptions, 'Preprocessing')

        self.textDetectionOptions = QWidget(self)
        self.paragraphsComboBox = QInputWithLabel('Paragraph detection', 'text_detection', QComboBox, combobox_list=[x.name for x in ParagraphDetectionTypes])
        self.textThresholdSlider = QInputWithLabel('Text threshold', 'text_detection', QSlider, min_val=0.1, max_val=1, step=0.01)
        self.linkThresholdSlider = QInputWithLabel('Link threshold', 'text_detection', QSlider, min_val=0.1, max_val=1, step=0.01)
        self.lowTextSlider = QInputWithLabel('Low text', 'text_detection', QSlider, min_val=0.1, max_val=1, step=0.01)

        self.textDetectionOptionsLayout = QVBoxLayout(self)
        self.textDetectionOptionsLayout.addWidget(self.paragraphsComboBox)
        self.textDetectionOptionsLayout.addWidget(self.textThresholdSlider)
        self.textDetectionOptionsLayout.addWidget(self.linkThresholdSlider)
        self.textDetectionOptionsLayout.addWidget(self.lowTextSlider)

        self.textDetectionOptionsLayout.addStretch()
        self.textDetectionOptions.setLayout(self.textDetectionOptionsLayout)
        self.optionsTabs.addTab(self.textDetectionOptions, 'Text Detection')

        self.mainLayout = QGridLayout(self)
        self.mainLayout.addWidget(self.optionsTabs, 0, 0, 1, 4)
        self.mainLayout.addWidget(self.startButton, 1, 1, 1, 2)
        self.container.setLayout(self.mainLayout)
        self.setCentralWidget(self.container)

        self.hotkey = keyboard.GlobalHotKeys({'<ctrl>+<shift>+t': self.toggle_overlay})
        self.hotkey.start()

    def closeEvent(self, event):
        self.hotkey.stop()
        self.overlay.close()
        if DEBUG:
            print({section: dict(CONFIG[section]) for section in CONFIG.sections()})
        save_config()
        QApplication.instance().quit()

    def toggle_overlay(self):
        if self.setupCompleted:
            if self.overlayStarted:
                self.startButton.setText('Start')
                self.generalOptions.setEnabled(True)
                self.preprocessingOptions.setEnabled(True)
                self.textDetectionOptions.setEnabled(True)
                self.overlay.stop_overlay()
            else:
                self.startButton.setText('Stop')
                self.generalOptions.setEnabled(False)
                self.preprocessingOptions.setEnabled(False)
                self.textDetectionOptions.setEnabled(False)
                settings = {'OCR': OCRSelect(int(CONFIG['general']['OCR method'])),
                            'translator': CONFIG['general']['translation method'],
                            'preprocess': CONFIG['preprocessing']['preprocess'] == 'True',
                            'resize': float(CONFIG['preprocessing']['resize']),
                            'blur_kernel_size': int(CONFIG['preprocessing']['gaussian blur']),
                            'equalize': CONFIG['preprocessing']['histogram equalization'] == 'True',
                            'binarization': BinarizationTypes(int(CONFIG['preprocessing']['binarization type'])),
                            'blocks': int(CONFIG['preprocessing']['binarization blocks']),
                            'c': int(CONFIG['preprocessing']['binarization c value']),
                            'binarization_blur_kernel_size': int(CONFIG['preprocessing']['post binarization gaussian blur']),
                            'paragraph_detection': ParagraphDetectionTypes(int(CONFIG['text_detection']['paragraph detection'])),
                            'text_threshold': float(CONFIG['text_detection']['text threshold']),
                            'link_threshold': float(CONFIG['text_detection']['link threshold']),
                            'low_text': float(CONFIG['text_detection']['low text']),
                            'min_japanese_characters': 1,
                            'font_size': float(CONFIG['general']['font size'])
                            }
                self.overlay.start_overlay(settings)
            self.overlayStarted = not self.overlayStarted

    def on_setup_completed(self, set_default_translator=False):
        self.startButton.setText('Start')
        self.startButton.setEnabled(True)
        self.setupCompleted = True
        if set_default_translator:
            self.translatorComboBox.set_value('Google')

    @staticmethod
    def get_available_translations():
        tl_list = ['Google']
        if CONFIG['general']['deepl api key'] != '':
            tl_list.append('DeepL')
        if os.path.exists(os.path.join(PATH, 'models/sugoi_v4model_ctranslate2/ct2Model')) and os.path.exists(os.path.join(PATH, 'models/sugoi_v4model_ctranslate2/spmModels')):
            tl_list.append('Sugoi')
        return tl_list


def setup_application(path, cuda=False, debug=False, args=None):
    global PATH
    global CONFIG_PATH
    global DEBUG
    print(platform)
    PATH = path
    CONFIG_PATH = os.path.join(path, 'config.ini')
    DEBUG = debug
    if debug:
        print(CONFIG_PATH)
    if os.path.isfile(CONFIG_PATH):
        CONFIG.read(CONFIG_PATH)
    else:
        create_default_config()
    if args is None:
        app = QApplication()
    else:
        app = QApplication(args)
    thread = QThread()
    thread.start()
    worker = TranslationWorker()
    worker.moveToThread(thread)
    master = Master(thread)
    master.tl_setup.connect(worker.setup)
    master.tl_start.connect(worker.verify_and_load_models)
    master.tl_tick.connect(worker.translate)
    overlay_window = OverlayWindow(thread_master=master, cuda=cuda)
    worker.tl_completed.connect(overlay_window.update_overlay)
    setup_window = SetupWindow(overlay=overlay_window)
    worker.setup_completed.connect(lambda x: setup_window.on_setup_completed(set_default_translator=x))
    setup_window.show()
    overlay_window.show()
    return app, overlay_window, setup_window, master, worker
