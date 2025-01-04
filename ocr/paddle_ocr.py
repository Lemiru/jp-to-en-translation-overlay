from paddleocr import PaddleOCR


class PaddleOcrBatchProcessing:
    def __init__(self, threshold=0.85):
        self.model = PaddleOCR(lang='japan', use_angle_cls=True, max_text_length=250)
        self.threshold = threshold

    def batch(self, images, recognition_only=False):
        if len(images) == 0:
            return []
        if not recognition_only:
            texts = [self.model.ocr(image)[0] if image.shape != (0, 0) else [[None, ('', 0)]] for image in images]
            texts_strings = []
            for text in texts:
                string = ''
                if text is not None:
                    for line in text:
                        if line[1][1] >= self.threshold:
                            string += line[1][0]
                texts_strings.append(string)
            return texts_strings
        else:
            texts = [self.model.ocr(image, det=False)[0] for image in images]
            return [text[0][0] if text[0][1] >= self.threshold else '' for text in texts]

    def single(self, image):
        return [x[1][0] if x[1][1] > self.threshold else '' for x in self.model.ocr(image)]
