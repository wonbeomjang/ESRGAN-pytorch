import cv2
import os
from multiprocessing import Pool


def crop_image(path, image_size, stride):
    image_list = os.listdir(path)

    for index, image_name in enumerate(image_list):
        if not image_name.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(path, image_name))
        height, width, channels = img.shape
        num_row = height // stride
        num_col = width // stride
        image_index = 0

        if index % 100 == 0:
            print(f'[*] [{index}/{len(image_list)}] Make patch {os.path.join(path, image_name)}')

        for i in range(num_row):
            if (i+1)*image_size > height:
                break
            for j in range(num_col):
                if (j+1)*image_size > width:
                    break
                cv2.imwrite(os.path.join(path, f'{image_name.split(".")[0]}_{image_index}.png'), img[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size])
                image_index += 1
        os.remove(os.path.join(path, image_name))


def make_patches():
    image_dir = [('hr', 128, 100), ('lr', 32, 25)]
    pool = Pool(processes=2)
    pool.starmap(crop_image, image_dir)
    pool.close()
    pool.join()
