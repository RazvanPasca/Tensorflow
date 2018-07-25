import argparse
import os

import numpy as np
from PIL import Image
from scipy.misc import imresize


def read_image(imagefile, dtype=np.float32):
    image_open = Image.open(imagefile)
    image = np.array(image_open, dtype=dtype)
    return imresize(image, 100)


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


def horizontal_flip(image, rate=0.9):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
        return image
    return None


def vertical_flip(image, rate=0.9):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
        return image
    return None


def crop_and_keep(image, scale_range, crop_size, rate=0.9):
    if np.random.rand() < rate:
        scale_size = np.random.randint(*scale_range)
        image = imresize(image, (scale_size, scale_size))
        image = random_crop(image, crop_size)
        return image
    return None


def change_gamma(image, rate=0.3):
    if np.random.rand() < rate:
        image_brighter = 255 * (image / 255) ** 0.5
        image_darker = 255 * (image / 255) ** 2
        return image_brighter, image_darker
    return None, None


def contrast_adjustment(image, rate=0.9):
    min_red = 255
    min_blue = 255
    min_green = 255
    max_red = 0
    max_blue = 0
    max_green = 0
    new_image_high = np.zeros((100,100,3))
    new_image_low = np.zeros((100,100,3))
    w, h, _ = image.shape
    if np.random.rand() < rate:
        for i in range(w):
            for j in range(h):
                pixel = image[i][j]
                min_red = min(pixel[0], min_red)
                min_green = min(pixel[1], min_green)
                min_blue = min(pixel[2], min_blue)
                max_red = max(pixel[0], max_red)
                max_green = max(pixel[1], max_green)
                max_blue = max(pixel[2], max_blue)
        for i in range(w):
            for j in range(h):
                pixel = image[i][j]
                new_pixel_r = adjust_histo(pixel[0], max_red, min_red)
                new_pixel_g = adjust_histo(pixel[1], max_green, min_green)
                new_pixel_b = adjust_histo(pixel[2], max_blue, min_blue)
                new_image_high[i][j] = (new_pixel_r, new_pixel_g, new_pixel_b)

                new_pixel_r = adjust_histo(pixel[0], max_red, min_red, False)
                new_pixel_g = adjust_histo(pixel[1], max_green, min_green, False)
                new_pixel_b = adjust_histo(pixel[2], max_blue, min_blue, False)
                new_image_low[i][j] = (new_pixel_r, new_pixel_g, new_pixel_b)
        return new_image_high, new_image_low
    return None, None


def adjust_histo(value_in, max_in, min_in, high=True):
    if high:
        return (value_in - min_in) * 255 / (max_in - min_in)
    else:
        return 30 + (value_in - min_in) * 195 / (max_in - min_in)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Data Augmentation')
    # parser.add_argument('infile', default='seven.png')
    parser.add_argument('--outdir', '-o', default='./images')
    args = parser.parse_args()

    for _ in range(2):
        for dirpath, dirs, files in os.walk("./lena"):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                inimg = read_image(fname)
                print(fname)
                inimg = resize(inimg, 100)
                # save_image(inimg, fname)
                image_brighter, image_darker = change_gamma(inimg)
                save_image(inimg, fname)
                image_c_brighter, image_c_darker = contrast_adjustment(inimg)
                save_image(
                    image_c_brighter,
                    os.path.join(dirpath, '{}_high_C.jpg'.format(filename)))
                save_image(
                    image_c_darker,
                    os.path.join(dirpath, '{}_low_C.jpg'.format(filename)))
                save_image(
                    image_brighter,
                    os.path.join(dirpath, '{}_bright_gamma.jpg'.format(filename)))
                save_image(
                    image_darker,
                    os.path.join(dirpath, '{}_dark_gamma.jpg'.format(filename)))
                save_image(
                    horizontal_flip(inimg),
                    os.path.join(dirpath, '{}_h_flip.jpg'.format(filename)))
                save_image(
                    vertical_flip(inimg),
                    os.path.join(dirpath, '{}_v_flip.jpg'.format(filename)))
                save_image(
                    crop_and_keep(resize(inimg,400), (256, 480), 150),
                    os.path.join(dirpath, '{}_cropped.jpg'.format(filename)))
