from typing import Union

import cv2
import numpy as np
import torch
from torch_kmeans import KMeans
from PIL import Image, ImageDraw, ImageFont

from common import BinarizationTypes

model = KMeans(n_clusters=4, verbose=False)


def check_for_image_difference(image1, image2, threshold=0.001):
    h, w = image1.shape[:2]
    img1 = cv2.GaussianBlur(image1, (7, 7), (7 - 1) / 6)
    img1 = cv2.Canny(img1, 150, 250)
    img2 = cv2.GaussianBlur(image2, (7, 7), (7 - 1) / 6)
    img2 = cv2.Canny(img2, 150, 250)
    diff = cv2.absdiff(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))
    return mse > threshold


def find_dominant_color(img):
    tensor = torch.from_numpy(img.reshape(1, -1, 3)).float()
    result = model(tensor)
    count = torch.bincount(result.labels[0])
    indice = count.argmax()
    return result.centers[0, indice].detach().numpy()


def create_new_image(dim: (int, int)):
    img = np.zeros((dim[0], dim[1], 4), dtype=np.uint8)
    return img


def draw_boxes(img_source, boxes, blur: Union[int, None] = None):
    new_image = create_new_image((img_source.shape[0], img_source.shape[1], 4))
    for box in boxes:
        if box[1] is None:
            bg_image = img_source[box[0][0][1]:box[0][1][1], box[0][0][0]:box[0][1][0]]
            color = np.array(find_dominant_color(bg_image), dtype=int).tolist()
            color = [*color, 255]
            cv2.rectangle(new_image, box[0][0], box[0][1], color, thickness=-1)
        else:
            for x in box[1]:
                bg_image = img_source[x[0][1]:x[1][1], x[0][0]:x[1][0]]
                color = np.array(find_dominant_color(bg_image), dtype=int).tolist()
                color = [*color, 255]
                cv2.rectangle(new_image, x[0], x[1], color, thickness=-1)
    if blur is not None:
        new_image = cv2.blur(new_image, (blur, blur))
    return new_image


def draw_text(img, boxes, texts, max_font_size=10.0, padding_x=0, padding_y=0):
    font = ImageFont.truetype("arial.ttf", max_font_size)
    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)
    for box, text in zip(boxes, texts):
        text = text.splitlines()
        words = []
        for line in text:
            words.extend(line.split())
            words.extend('\n')
        words.pop()
        multiline_text = None
        for word in words:
            if multiline_text is None:
                multiline_text = word
                continue
            temp = multiline_text + (' ' + word)
            bbox = draw.multiline_textbbox((box[0][0][0] + padding_x, box[0][0][1] + padding_y), temp, font=font)
            if bbox[2] > box[0][1][0]:
                multiline_text += '\n' + word
            else:
                multiline_text = temp
        draw.multiline_text((box[0][0][0] + padding_x, box[0][0][1] + padding_y), multiline_text, stroke_width=1, stroke_fill='black', font=font)
    return np.asarray(pil_image)


def process_for_ocr(img, block: int = 3, c: int = 0, scale: float = 2.0, gaussian_kernel: int = 1, gaussian_kernel_post: int = 1, binarization: BinarizationTypes = BinarizationTypes.NoBinarization, equalize: bool = False):
    size_x = int(img.shape[1] * scale)
    size_y = int(img.shape[0] * scale)
    img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if equalize:
        img = cv2.equalizeHist(img)
    if gaussian_kernel != 1:
        img = cv2.GaussianBlur(img, (gaussian_kernel, gaussian_kernel), (gaussian_kernel-1)/6)
    match binarization:
        case BinarizationTypes.NoBinarization:
            return img
        case BinarizationTypes.Normal:
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)
        case BinarizationTypes.Inverted:
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c)
        case BinarizationTypes.NormalWithNegC:
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -c)
        case BinarizationTypes.InvertedWithNegC:
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, -c)
    if gaussian_kernel_post != 1:
        img = cv2.GaussianBlur(img, (gaussian_kernel_post, gaussian_kernel_post), (gaussian_kernel_post-1)/6)
    return img
