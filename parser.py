from concurrent.futures import as_completed, Executor
from itertools import chain

import cv2
import numpy as np

from config import template_dict as td, template_image_dict as tid, masks


__all__ = ['extract_metadata']


def _debug_image(im: cv2.Mat) -> None:
    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _desc(selected_desc):
    desc = ''
    if 'crit' in selected_desc:
        if 'chance' in selected_desc:
            desc = '치명타 확률'
        elif 'damage' in selected_desc:
            desc = '치명타 피해'
    elif 'effect' in selected_desc:
        if 'resist' in selected_desc:
            desc = '효과 저항'
        elif 'hit' in selected_desc:
            desc = '효과 적중'
    elif 'speed' in selected_desc:
        desc = '속도'
    elif 'attack' in selected_desc:
        desc = '공격력'
    elif 'defence' in selected_desc:
        desc = '방어력'
    elif 'health' in selected_desc:
        desc = '체력'

    return desc

def _detect(image: cv2.Mat, temp_path: str, tag: dict, THRESHOLD: float = 0.90) -> dict:
    retval = {}
    template = tid[temp_path]
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= THRESHOLD)
    for _ in zip(*locations[::-1]):
        if 'mainoption' in tag:
            retval |= {'mainoption': {_desc(tag['mainoption'])}}
        else:
            retval |= tag
    return retval


def _detect_mop(desc_im: cv2.Mat, option_text: dict, THRESHOLD: float = 0.90) -> dict:
    pass
    selected_desc = set()
    # desc_im
    for desc_path, desc in option_text.items():
        template = tid[desc_path]
        assert template is not None

        result = cv2.matchTemplate(desc_im, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= THRESHOLD)
        for _ in zip(*locations[::-1]):
            selected_desc.add(desc)
    return {'mainoption': _desc(selected_desc)}


def _detect_sop(desc_im: cv2.Mat, num_im: cv2.Mat, op: str, option_text: dict, option_num: dict, THRESHOLD: float = 0.90) -> dict:
    num_text = ''
    num_locations = []
    
    desc = _detect_mop(desc_im, option_text)['mainoption']

    for num_path, num in option_num.items():
        template = tid[num_path]
        assert template is not None
        dy, dx = template.shape[0:2]

        detected = False
        run_first = True
        
        while run_first or detected:
            run_first = False
            detected = False
            result = cv2.matchTemplate(num_im, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= THRESHOLD)
            for tx, ty in zip(*locations[::-1]):
                detected = True
                num_locations.append((tx, num))
                cv2.rectangle(num_im, (tx, ty), (tx+dx, ty+dy), (255, 255, 255), -1)
                break

    for _, num in sorted(num_locations):
        num_text += num
        
    return {op: {desc: num_text}}


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
    temp = {}
    fs = []

    # item_fullname
    for i, tag in list(td['part_icon'].items()) + list(td['part_set'].items()):
        im = _apply_crop(image.copy(), masks['item_fullname'])
        fs.append(ex.submit(_detect, im, i, tag))

    # item_portrait
    for i, tag in td['lock'].items():
        im = _apply_crop(image.copy(), masks['item_portrait'])
        fs.append(ex.submit(_detect, im, i, tag, 0.9))

    for i, tag in td['reinforce'].items():
        im = _apply_crop(image.copy(), masks['item_portrait'])
        fs.append(ex.submit(_detect, im, i, tag, 0.85))

    # options
    for op in ['mainoption', 'suboption_1', 'suboption_2', 'suboption_3', 'suboption_4']:
        desc_im = _apply_crop(image.copy(), masks[f'{op}_text'])
        if op == 'mainoption':
            fs.append(ex.submit(_detect_mop, desc_im, td['option_text']))
        else:
            num_im = _apply_crop(image.copy(), masks[f'{op}_num'])
            fs.append(ex.submit(_detect_sop, desc_im, num_im, op, td['option_text'], td['option_num']))

    for f in as_completed(fs):
        temp.update(f.result())
    
    retval['part'] = temp.get('part', '')
    retval['set'] = temp.get('set', '')
    retval['type'] = temp.get('type', '')
    retval['lock'] = temp.get('lock', None)
    retval['mainoption'] = temp.get('mainoption', '')
    retval['reinforce'] = temp.get('reinforce', '')
    retval['suboption'] = [
        temp.get('suboption_1', ''),
        temp.get('suboption_2', ''),
        temp.get('suboption_3', ''),
        temp.get('suboption_4', '')
    ]

    return retval
