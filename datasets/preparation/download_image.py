import os
import shutil
from preparation.utils import download_url, unzip_zip_file, unzip_tar_file
from glob import glob


def download_dataset():
    DIV2K_HR = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    DIV2K_LR = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
    FLICKR2K = "http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"

    if not os.path.exists('temp'):
        os.makedirs('temp')

    if not os.path.exists(os.path.join('hr')):
        os.makedirs(os.path.join('hr'))
    if not os.path.exists(os.path.join('lr')):
        os.makedirs(os.path.join('lr'))

    download_url(DIV2K_HR, os.path.join('temp', 'DIV2K_HR.zip'))
    download_url(DIV2K_LR, os.path.join('temp', 'DIV2K_LR.zip'))
    download_url(FLICKR2K, os.path.join('temp', 'FLICKR2K.tar'))

    print('[!] Upzip zipfile')
    unzip_zip_file(os.path.join('temp', 'DIV2K_HR.zip'), 'temp')
    unzip_zip_file(os.path.join('temp', 'DIV2K_LR.zip'), 'temp')
    unzip_tar_file(os.path.join('temp', 'FLICKR2K.tar'), 'temp')

    print('[!] Reformat DIV2K HR')
    image_path = glob('temp/DIV2K_train_HR/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('hr', f'{index:04d}.png'))

    print('[!] Reformat DIV2K LR')
    image_path = glob('temp/DIV2K_train_LR_bicubic/X4/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('lr', f'{index:04d}.png'))

    print('[!] Reformat FLICKR2K HR')
    image_path = glob('temp/Flickr2K/Flickr2K_HR/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('hr', f'{index:05d}.png'))

    print('[!] Reformat FLICKR2K LR')
    image_path = glob('temp/Flickr2K/Flickr2K_LR_bicubic/X4/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('lr', f'{index:05d}.png'))

    shutil.rmtree('temp')
