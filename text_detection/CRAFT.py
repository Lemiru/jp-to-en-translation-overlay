from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.backends import cudnn
from torch.autograd import Variable

from models.craft import craft, refinenet, imgproc, craft_utils
from common import ParagraphDetectionTypes


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class CRAFTModel:
    def __init__(self,
                 refiner=False,
                 cuda=False,
                 square_size=1280,
                 mag_ratio=1.5,
                 ):
        self.refiner = refiner
        self.cuda = cuda
        self.square_size = square_size
        self.mag_ratio = mag_ratio
        self.craftModel = craft.CRAFT()
        self.craftModel.load_state_dict(copyStateDict(torch.load("models/craft/pretrained/craft_mlt_25k.pth")))
        if cuda:
            self.craftModel = self.craftModel.cuda()
            self.craftModel = torch.nn.DataParallel(self.craftModel)
            cudnn.benchmark = False
        self.craftModel.eval()

        if refiner:
            self.craft_refiner = refinenet.RefineNet()
            self.craft_refiner.load_state_dict(copyStateDict(torch.load("models/craft/pretrained/craft_refiner_CTW1500.pth")))
            if cuda:
                self.craft_refiner = self.craft_refiner.cuda()
                self.craft_refiner = torch.nn.DataParallel(self.craft_refiner)
            self.craft_refiner.eval()

    def detect(self, img, text_threshold, link_threshold, low_text):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, square_size=self.square_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        with torch.no_grad():
            y, feature = self.craftModel(x)
            score_text = y[0, :, :, 0].cpu().data.numpy()
            score_link = y[0, :, :, 1].cpu().data.numpy()
            if self.refiner:
                y_refiner = self.craft_refiner(y, feature)
                score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold=text_threshold,
                                               link_threshold=link_threshold, low_text=low_text, poly=False)
        return craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)


def detect_paragraphs(boxes: np.ndarray, max_distance_x=10, max_distance_y=10, vertical=False):
    paragraphs = []
    exclude = []
    for x in range(len(boxes)):
        left_x, top_x = boxes[x][0]
        right_x, bottom_x = boxes[x][1]
        current_paragraph = [x]
        for y in range(len(boxes)):
            if (x == y) or (y in exclude):
                continue
            left_y, top_y = boxes[y][0]
            right_y, bottom_y = boxes[y][1]
            if vertical:
                inside = top_x < top_y < bottom_x and left_x < right_y < right_x
                if abs(top_x - top_y) < max_distance_y:
                    if 0 <= -(right_y - left_x) < max_distance_x or inside or left_x < right_y < right_x:
                        if x not in exclude:
                            exclude.append(x)
                        exclude.append(y)
                        current_paragraph.append(y)
                        left_x = left_y
                        bottom_x = max(bottom_x, bottom_y)
            else:
                inside = top_x < top_y < bottom_x and left_x < left_y < right_x
                if abs(left_x - left_y) < max_distance_x:
                    if 0 <= (top_y - bottom_x) < max_distance_y or inside or top_x < top_y < bottom_x:
                        if x not in exclude:
                            exclude.append(x)
                        exclude.append(y)
                        current_paragraph.append(y)
                        bottom_x = bottom_y
                        right_x = max(right_x, right_y)
        if len(current_paragraph) > 1:
            paragraphs.append(current_paragraph)

    return paragraphs


def calculate_paragraph_boxes(paragraphs, boxes):
    paragraph_boxes = []
    for paragraph in paragraphs:
        left = min(boxes[x][0][0] for x in paragraph)
        top = min(boxes[x][0][1] for x in paragraph)
        right = max(boxes[x][1][0] for x in paragraph)
        bottom = max(boxes[x][1][1] for x in paragraph)
        paragraph_boxes.append([[[left, top], [right, bottom]], [boxes[x] for x in paragraph]])
    return paragraph_boxes


def prepare_for_text_detection(boxes, image_size, paragraph: ParagraphDetectionTypes = ParagraphDetectionTypes.NoDetection):
    if len(boxes) == 0:
        return []
    new_boxes = []
    boxes = np.array(boxes, dtype=np.int32)
    for box in boxes:
        x1, y1 = max(min(box[0][0], box[1][0], box[2][0], box[3][0]), 0), max(min(box[0][1], box[1][1], box[2][1], box[3][1]), 0)
        x2, y2 = min(max(box[0][0], box[1][0], box[2][0], box[3][0]), image_size[1]), min(max(box[0][1], box[1][1], box[2][1], box[3][1]), image_size[0])
        new_box = [[x1 if x1 < x2 else x2, y1 if y1 < y2 else y2], [x2 if x1 < x2 else x1, y2 if y1 < y2 else y1]]
        new_boxes.append(new_box)
    new_boxes = np.array(new_boxes)
    boxes_in_paragraph = []
    if paragraph != ParagraphDetectionTypes.NoDetection:
        if paragraph == ParagraphDetectionTypes.Horizontal:
            paragraphs = detect_paragraphs(new_boxes, max_distance_x=20, max_distance_y=20)
        elif paragraph == ParagraphDetectionTypes.Vertical:
            right_coordinates = new_boxes[:, 1, 0]
            sorted_right_coordinates = np.argsort(-right_coordinates)
            new_boxes = new_boxes[sorted_right_coordinates]
            paragraphs = detect_paragraphs(new_boxes, max_distance_x=20, max_distance_y=20, vertical=True)
        else:
            raise ValueError("Invalid paragraph detection type")

        for paragraph in paragraphs:
            for box in paragraph:
                if box not in boxes_in_paragraph:
                    boxes_in_paragraph.append(box)
        paragraph_boxes = calculate_paragraph_boxes(paragraphs, new_boxes)
        final_boxes = []
        for x in range(len(new_boxes)):
            if x not in boxes_in_paragraph:
                final_boxes.append([new_boxes[x], None])
        final_boxes = final_boxes + paragraph_boxes
        return final_boxes
    else:
        return [[new_box, None] for new_box in new_boxes]

