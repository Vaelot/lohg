import cv2
import glob

part_icon_path = './templates/parts/part_icon/*.png'
part_icon_pos = (1250, 156, 1292, 196)

part_set_path = './templates/parts/part_set/*.png'
part_set_pos = (1294,156,1385,197)


def crop():
    # crop
    for i in glob.glob(part_set_path):
        img = cv2.imread(i)
        crop_img = img[part_set_pos[1]:part_set_pos[3], part_set_pos[0]:part_set_pos[2]]
        cv2.imwrite(i, crop_img)

if __name__ == '__main__':
    crop()
