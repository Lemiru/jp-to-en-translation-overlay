import re
import jaconv
import cv2

from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, GenerationMixin


class MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    pass


class MangaOcrBatchProcessing:
    def __init__(self, cuda=False):
        self.processor = ViTImageProcessor.from_pretrained("kha-white/manga-ocr-base")
        self.tokenizer = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")
        self.model = MangaOcrModel.from_pretrained("kha-white/manga-ocr-base")
        if cuda:
            self.model.cuda()

    def batch(self, images):
        images_gray = [cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) if len(x.shape) != 2 else
                       cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) for x in images]
        images = self.processor(images_gray, return_tensors="pt").pixel_values.squeeze()
        results = self.model.generate(images.to(self.model.device), max_length=256).cpu()
        decoded = [post_process(self.tokenizer.decode(x, skip_special_tokens=True)) for x in results]
        return decoded

    def single(self, image):
        image_gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) if len(image.shape) != 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.processor(image_gray, return_tensors="pt").pixel_values.squeeze()
        results = self.model.generate(image[None].to(self.model.device))[0].cpu()
        decoded = self.tokenizer.decode(results, skip_special_tokens=True)
        decoded = post_process(decoded)
        return decoded

    def separate_per_paragraph(self, paragraph_boxes, full_image):
        images = []
        paragraph_idx = []
        count = 0
        for x in paragraph_boxes:
            if x[1] is None:
                images.append(full_image[x[0][0][1]:x[0][1][1], x[0][0][0]:x[0][1][0]])
                paragraph_idx.append(count)
            else:
                for y in x[1]:
                    images.append(full_image[y[0][1]:y[1][1], y[0][0]:y[1][0]])
                    paragraph_idx.append(count)
            count += 1
        if len(images) == 1:
            texts = [self.single(images[0])]
        else:
            texts = self.batch(images)
        grouped_texts = ['' for i in range(count)]
        for x, y in zip(texts, paragraph_idx):
            grouped_texts[y] = grouped_texts[y] + x + ' '
        return grouped_texts


def post_process(text):
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text
