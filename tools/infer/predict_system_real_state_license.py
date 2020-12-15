# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import tools.infer.utility as utility
from ppocr.utils.utility import initial_logger

logger = initial_logger()
import cv2
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_cls as predict_cls
import copy
import numpy as np
import math
import time
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from PIL import Image
from tools.infer.utility import draw_ocr
from tools.infer.utility import draw_ocr_box_txt

import re
import json


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            print("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)
        print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def get_nearest_left_box(tbox, tidx, boxes, txts, debug=False):
    left_thres = 15  # TODO should tune
    above_below_thres = 0  # TODO should tune
    nearest_idx = -1
    min_dis = -1
    tbox_height = math.sqrt((tbox[0][0] - tbox[3][0])**2 + (tbox[0][1] - tbox[3][1])**2)
    tbox_width = math.sqrt((tbox[0][0] - tbox[1][0])**2 + (tbox[0][1] - tbox[1][1])**2)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if tidx == idx:
            continue
        if '5i5j' in txt or '5151' in txt or '5i5' in txt or '5i' in txt or '5j' in txt:
            continue
        if box[1][0]-left_thres > tbox[0][0]:
            if debug:
                print('box[1][0]-left_thres > tbox[0][0]:', box[1][0], tbox[0][0])
            continue
        if abs(box[1][1] - tbox[0][1])-above_below_thres >= tbox_height:
            if debug:
                print('abs(box[1][1] - tbox[0][1])-above_below_thres >= tbox_height:', abs(box[1][1] - tbox[0][1]), tbox_height)
            continue
        dis = math.sqrt((box[1][0] - tbox[0][0])**2+(box[1][1] - tbox[0][1])**2)
        if min_dis == -1 or dis < min_dis:
            min_dis = dis
            nearest_idx = idx
    return nearest_idx


def get_nearest_right_box(tbox, tidx, boxes, txts, debug=False):
    right_thres = 15  # TODO should tune
    above_below_thres = 0  # TODO should tune
    nearest_idx = -1
    min_dis = -1
    tbox_height = math.sqrt((tbox[0][0] - tbox[3][0])**2 + (tbox[0][1] - tbox[3][1])**2)
    tbox_width = math.sqrt((tbox[0][0] - tbox[1][0])**2 + (tbox[0][1] - tbox[1][1])**2)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if tidx == idx:
            continue
        if '5i5j' in txt or '5151' in txt or '5i5' in txt or '5i' in txt or '5j' in txt:
            continue
        if box[0][0]+right_thres < tbox[1][0]:
            if debug:
                print('box[0][0]+right_thres < tbox[1][0]:', box[0][0], tbox[1][0])
            continue
        if abs(box[0][1] - tbox[1][1])-above_below_thres >= tbox_height:
            if debug:
                print('abs(box[0][1] - tbox[1][1])-above_below_thres >= tbox_height:', abs(box[0][1] - tbox[1][1]), tbox_height)
            continue
        dis = math.sqrt((box[0][0] - tbox[1][0])**2+(box[0][1] - tbox[1][1])**2)
        if min_dis == -1 or dis < min_dis:
            min_dis = dis
            nearest_idx = idx
    return nearest_idx

def parse_real_estate_license(image, boxes, txts, scores=None, drop_score=0.5, font_path="./doc/simfang.ttf"):
    result = {'code': 0}
    h, w = image.height, image.width
    ownerName = ''
    address = ''
    realEstateNo = ''
    houseProperty = ''
    area = 0
    period = ''
    totalFloor = 0
    certNo = ''
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue

        if '5i5j' in txt or '5151' in txt or '5i5' in txt or '5i' in txt or '5j' in txt:
            continue
        
        # box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]

        neareast_right_idx = get_nearest_right_box(box, idx, boxes, txts)

        if '权利人' in txt:  # TODO dirty code
            if txt == '权利人':
                if neareast_right_idx == -1:
                    continue
                ownerName = txts[neareast_right_idx]
            else:
                ownerName = txt.replace('权利人', '')
        elif txt == '坐落' or txt == '落':
            if neareast_right_idx == -1:
                continue
            address = txts[neareast_right_idx]
        elif txt == '不动产单元号':
            if neareast_right_idx == -1:
                continue
            realEstateNo = txts[neareast_right_idx]
            neareast_right_right_idx = get_nearest_right_box(boxes[neareast_right_idx], neareast_right_idx, boxes, txts)
            if neareast_right_right_idx != -1:
                realEstateNo += txts[neareast_right_right_idx]
            neareast_right_right_right_idx = get_nearest_right_box(boxes[neareast_right_right_idx], neareast_right_right_idx, boxes, txts)
            if neareast_right_right_right_idx != -1:
                realEstateNo += txts[neareast_right_right_right_idx]
        elif txt == '权利性质':
            if neareast_right_idx == -1:
                continue
            houseProperty = txts[neareast_right_idx]
        elif txt == '面积' or txt == '积':
            if neareast_right_idx == -1:
                continue
            area_str = txts[neareast_right_idx]
            area_re = re.findall(r".*建筑面积(.+)[m|平].*", area_str)
            if len(area_re) > 0:
                area = float(area_re[0])
        elif txt == '使用期限':
            if neareast_right_idx == -1:
                continue
            period = txts[neareast_right_idx]
        elif '房屋总层数' in txt:
            totalFloor_str = txt
            print('totalFloor_str:', totalFloor_str)
            totalFloor_re = re.findall(r".*总层数[^\d*](\d+).*", totalFloor_str)
            print('totalFloor_re:', totalFloor_re)
            if len(totalFloor_re) > 0:
                totalFloor = int(totalFloor_re[0])
        elif '不动产权' in txt:
            certNo = txt
            if neareast_right_idx != -1:
                certNo += txts[neareast_right_idx]
            neareast_left_idx = get_nearest_left_box(box, idx, boxes, txts)
            if neareast_left_idx != -1:
                certNo = txts[neareast_left_idx]+certNo
            neareast_left_left_idx = get_nearest_left_box(boxes[neareast_left_idx], neareast_left_idx, boxes, txts)
            if neareast_left_left_idx != -1:
                certNo = txts[neareast_left_left_idx]+certNo
            certNo = certNo.replace('（', '').replace('）', '')
            if not certNo.endswith('号'):
                certNo += '号'
    data = {}
    data['ownerName'] = ownerName
    data['address'] = address
    data['realEstateNo'] = realEstateNo
    data['houseProperty'] = houseProperty
    data['area'] = area
    data['period'] = period
    data['totalFloor'] = totalFloor
    data['certNo'] = certNo
    result['data'] = data
    return result

def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
            # # TODO simple rotate method, should change
            # h, w = img.shape[:2]
            # center = (w // 2, h // 2)
            # if w > h:
            #     M_2 = cv2.getRotationMatrix2D(center, -90, 1)
            #     img = cv2.warpAffine(img, M_2, (w, h))
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))

        drop_score = 0.5
        dt_num = len(dt_boxes)
        for dno in range(dt_num):
            text, score = rec_res[dno]
            if score >= drop_score:
                text_str = "%s, %.3f" % (text, score)
                print(text_str)

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            json_result = parse_real_estate_license(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            print('json_result:', json_result)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            json.dump(json_result, open(os.path.join(draw_img_save, os.path.basename(image_file)+'.json'), 'w'), ensure_ascii=False)
            print("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))


if __name__ == "__main__":
    main(utility.parse_args())
