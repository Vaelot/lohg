import cv2
import numpy as np
import mss
import pytesseract
import keyboard
import mouse

# Tesseract 실행 파일 경로 설정 (설치된 경로로 변경)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# OCR을 수행할 영역 (전체 화면 기준 좌표)
capture_regions = {
    "mainop": { "top": 253, "left": 1470, "width": 270, "height": 32, "lang": "kor", "f": 1.5, "inv": False },
    # "reinforce": { "top": 239, "left": 1275, "width": 54, "height": 33, "lang": "eng", "f": 1.5, "inv": True },
    "subop_text1": { "top": 389, "left": 1272, "width": 228, "height": 39, "lang": "kor", "f": 1.5, "inv": False },
    "subop_num1": { "top": 389, "left": 1543, "width": 202, "height": 39, "lang": "eng", "f": 1.5, "inv": False },
    "subop_text2": { "top": 427, "left": 1272, "width": 228, "height": 39, "lang": "kor", "f": 1.5, "inv": False },
    "subop_num2": { "top": 427, "left": 1543, "width": 202, "height": 39, "lang": "eng", "f": 1.5, "inv": False },
    "subop_text3": { "top": 465, "left": 1272, "width": 228, "height": 39, "lang": "kor", "f": 1.5, "inv": False },
    "subop_num3": { "top": 465, "left": 1543, "width": 202, "height": 39, "lang": "eng", "f": 1.5, "inv": False },
    "subop_text4": { "top": 503, "left": 1272, "width": 228, "height": 39, "lang": "kor", "f": 1.5, "inv": False },
    "subop_num4": { "top": 503, "left": 1543, "width": 202, "height": 39, "lang": "eng", "f": 1.5, "inv": False },
    "part_text": { "top": 155, "left": 1290, "width": 350, "height": 45, "lang": "kor", "f": 1, "inv": False }

}
def capture_screen(top, left, width, height, **kwargs):
    """전체 화면을 캡처한 후 OCR을 수행할 영역을 크롭"""
    with mss.mss() as sct:
        full_screenshot = sct.grab(sct.monitors[2])  # 필요 시 sct.monitors[2]로 변경
        full_img = np.array(full_screenshot)
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2GRAY)

        # OCR 영역 크롭
        cropped_img = full_img[top:top+height, left:left+width]
        return cropped_img

def preprocess_image(image, **kwargs):
    """OCR 인식을 위한 이미지 전처리"""
    image = cv2.resize(image, None, fx=kwargs["f"], fy=kwargs["f"], interpolation=cv2.INTER_LANCZOS4)  # 확대
    image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # 이진화
    # if kwargs["inv"]:
    #     image = cv2.blur(image, ksize=(3, 3), borderType=cv2.BORDER_CONSTANT)
    # 배경은 흰색 (255)
    height, width = max(1600, image.shape[0]), max(1600, image.shape[1])  # 배경을 설정할 크기
    background = np.ones((height, width), dtype=np.uint8) * 255  # 흰색 배경 생성
    
    # 기존 이미지를 중앙에 붙이기 위한 좌표 계산
    y_offset = (height - image.shape[0]) // 2
    x_offset = (width - image.shape[1]) // 2
    
    # 배경에 기존 이미지를 중앙 배치
    background[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

    cv2.imshow("Cropped Region", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return background

def recognize_text(image, **kwargs):
    """OCR을 사용하여 텍스트 인식"""
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(image, config=custom_config, lang=kwargs["lang"])
    return extracted_text.strip()

def on_click():
    """Ctrl 키가 눌린 상태에서 마우스 좌클릭이 감지되면 OCR 실행"""
    if keyboard.is_pressed("ctrl"):
        print("[INFO] Ctrl + 좌클릭 감지됨, OCR 실행 중...")
        for key, v in capture_regions.items():
            captured_img = capture_screen(**v)
            processed_img = preprocess_image(captured_img, **v)
            recognized_text = recognize_text(processed_img, **v)
            print(f"{key} 인식된 텍스트:", recognized_text)

# 마우스 좌클릭 이벤트 감지
mouse.on_click(lambda: on_click())

print("[INFO] Ctrl + 마우스 좌클릭을 하면 OCR이 실행됩니다.")
keyboard.wait("esc")  # ESC 키를 누르면 프로그램 종료
