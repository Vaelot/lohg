import cv2
import glob

target = 'replica'
num_path = './templates/reinforce_{}/*.png'
lock_path = './templates/lock_{}/*.png'

num_pos = (1279, 243, 1330, 271)
#lock_pos = (1381, 243, 1424, 281)
lock_pos = (1381, 243, 1414, 281)


def crop():
    # crop
    for i in glob.glob(lock_path.format(target)):
        img = cv2.imread(i)
        crop_img = img[lock_pos[1]:lock_pos[3], lock_pos[0]:lock_pos[2]]
        cv2.imwrite(i, crop_img)
    for i in glob.glob(num_path.format(target)):
        img = cv2.imread(i)
        crop_img = img[num_pos[1]:num_pos[3], num_pos[0]:num_pos[2]]
        cv2.imwrite(i, crop_img)


if __name__ == '__main__':
    crop()
