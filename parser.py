from concurrent.futures import as_completed, Executor
from itertools import chain

import cv2
import numpy as np

from config import template_dict as td, masks


__all__ = ['extract_metadata']


THRESHOLD = 0.90


def _detect(image: cv2.Mat, temp_path: str, tag: dict) -> dict:
    retval = {}
    template = cv2.imread(temp_path)
    if template is not None:
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= THRESHOLD)
        for _ in zip(*locations[::-1]):
            retval |= tag
    else:
        print(f"Warning: template image is not found: {temp_path}")
    return retval


def _detect_order(image: cv2.Mat, temp_path: str, numbers: dict) -> dict:
    retval = {}
    num_locations = []
    for num_path, num in numbers.items():
        template = cv2.imread(num_path)
        assert template is not None

        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= THRESHOLD)
        for i in zip(*locations[::-1]):
            print(i)
            # retval |= tag
    return retval


def _apply_masks(image, masks) -> cv2.Mat:
    """ 여러 개의 마스크를 순차적으로 적용하는 함수 """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for x1, y1, x2, y2 in masks:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def _apply_crop(image, crop_coord) -> cv2.Mat:
    return image[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]]


def extract_metadata(ex: Executor, image: cv2.Mat) -> dict:
    retval = {}
    fs = []

    # item_fullname
    for i, tag in chain(td['part_icon'].items(), td['part_set'].items()):
        im = _apply_crop(image.copy(), masks['item_fullname'])
        fs.append(ex.submit(_detect, im, i, tag))

    # item_portrait
    for i, tag in chain(td['lock'].items(), td['reinforce'].items()):
        im = _apply_crop(image.copy(), masks['item_portrait'])
        fs.append(ex.submit(_detect, im, i, tag))

    for op in ['mainoption', 'suboption_1', 'suboption_2', 'suboption_3', 'suboption_4']:
        for i, tag in td['option_text'].items():
            im = _apply_crop(image.copy(), masks[f'{op}_text'])
            fs.append(ex.submit(_detect, im, i, tag))

        for i, tag in td['option_num'].items():
            im = _apply_crop(image.copy(), masks[f'{op}_num'])
            fs.append(ex.submit(_detect_order, im, i, td['option_num']))
        
    

    for f in as_completed(fs):
        retval.update(f.result())

    return retval
