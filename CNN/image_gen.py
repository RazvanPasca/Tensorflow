import argparse
import os

import numpy as np
from PIL import Image
from scipy.misc import imresize


def read_image(imagefile, dtype=np.float32):
    image = np.array(Image.open(imagefile), dtype=dtype)
    return image


def save_image(image, imagefile, data_format='channel_last'):
    if image is not None:
        image = np.asarray(image, dtype=np.uint8)
        image = Image.fromarray(image)
        image.save(imagefile)


def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


def normalize(image):
    image = image / 255
    return image


def resize(image, size):
    size = check_size(size)
    image = imresize(image, size)
    return image


def random_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def horizontal_flip(image, rate=0.1):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
        return image
    return None


def vertical_flip(image, rate=0.15):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
        return image
    return None


def crop_and_keep(image, scale_range, crop_size, rate=0.2):
    if np.random.rand() < rate:
        scale_size = np.random.randint(*scale_range)
        image = imresize(image, (scale_size, scale_size))
        image = random_crop(image, crop_size)
        return image
    return None


def change_gamma(image, rate=0.2):
    if np.random.rand() < rate:
        image_brighter = 255 * (image / 255) ** 0.5
        image_darker = 255 * (image / 255) ** 2
        return image_brighter, image_darker
    return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Data Augmentation')
    parser.add_argument('infile', default='seven.png')
    parser.add_argument('--outdir', '-o', default='./images')
    args = parser.parse_args()

    for _ in range(2):
        for dirpath, dirs, files in os.walk(args.infile):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                inimg = read_image(fname)
                print(fname)
                inimg = resize(inimg, 100)
                save_image(inimg,fname)
                # image_brighter, image_darker = change_gamma(inimg)
                # save_image(
                #     image_brighter,
                #     os.path.join(dirpath, '{}_bright_gamma.jpg'.format(filename)))
                # save_image(
                #     image_darker,
                #     os.path.join(dirpath, '{}_dark_gamma.jpg'.format(filename)))
                # save_image(
                #     horizontal_flip(inimg),
                #     os.path.join(dirpath, '{}_h_flip.jpg'.format(filename)))
                # save_image(
                #     vertical_flip(inimg),
                #     os.path.join(dirpath, '{}_v_flip.jpg'.format(filename)))
                # save_image(
                #     crop_and_keep(resize(inimg,400), (256, 480), 150),
                #     os.path.join(dirpath, '{}_cropped.jpg'.format(filename)))
