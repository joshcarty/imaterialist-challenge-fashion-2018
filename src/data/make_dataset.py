"""
Module for downloading training, validation and
test images for the iMaterialist Challenge (Fashion)
at FGVC5 Kaggle challenge.

Uses threading from concurrent.futures and a progress bar from
tqdm.

https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/data
"""
import os
import io
import json
import argparse

import requests
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image


def get_image(url, path, resize=256):
    """
    Function to download and write an image to disk.

    Args:
        url (str): URL of iamge.
        path( str): Path to save image.

    Returns:
        None
    """
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))

    if resize:
        img = resize_image(img, resize)

    img.save(path)
    

def resize_image(img, size=256):
    """
    Resize images so that the longest edge is at most
    `size` pixels.

    Args:
        img (PIL.Image): Image to resize.
        size (int): Maximum dimension of the image.

    TODO (@joshcarty):
        Surely there is a nicer way.
    """
    max_edge = np.max(img.size)
    i_max_edge = np.argmax(img.size)
    i_min_edge = 1 - i_max_edge

    new_shape = [0, 0]
    new_shape[i_max_edge] = size

    scale = size / max_edge

    new_shape[i_min_edge] = int(scale * min(img.size))
    new_shape = tuple(new_shape)

    return img.resize(new_shape, Image.BICUBIC)


def load_urls(path):
    """
    Load URLs and filenames from JSON files

    Args:
        path (str): Path to data file.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    return data['images']


def dispatch(iterable):
    for _ in iterable:
        pass
    

def get_images(url_ids, dirpath, prefix=None, max_workers=100, n=None,
               skip_exists=True, resize=256):
    """
    Gets images from list of dicts and outputs
    them to path.

    Args:
        url_ids (list): List of dicts with urls and file ids.
        dirpath (str): Path to output directory.
        max_workers (int): Maximum number of threads to download with.
        n (int): Maximum number of images to download.
    """

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = []

        url_ids = url_ids[:n] if n else url_ids

        for pair in url_ids:
            url = pair['url']

            if prefix:
                filename = '{}_{}.jpg'.format(prefix, pair['imageId'])
            else:
                filename = '{}.jpg'.format(pair['imageId'])

            path = os.path.join(dirpath, filename)

            if skip_exists and not os.path.exists(path):
                futures.append(executor.submit(get_image, url, path))

        
    dispatch(tqdm(as_completed(futures), total=len(futures),
                  unit='image', unit_scale=True, leave=True,
                  desc=dirpath))


def make_dataset(dataset):
    """
    Creates train, validation and test folders
    in dataset directory if they do not already
    exist.

    Args:
        dataset (str): Path to dataset.
    """

    for folder in ('train', 'validation', 'test'):
        if not os.path.exists(os.path.join(dataset, folder)):
            os.makedirs(os.path.join(dataset, folder))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--max_workers', default=100, type=int)
    parser.add_argument('--max_images', default=None, type=int)
    parser.add_argument('--resize', default=256, type=int)
    parser.add_argument('--skip_exists', default=True, type=bool)

    return parser.parse_args()


def main():
    args = parse_args()

    make_dataset(args.dataset)

    train = load_urls(args.train)
    validation = load_urls(args.validation)
    test = load_urls(args.test)

    get_images(train, os.path.join(args.dataset, 'train'),
               max_workers=args.max_workers, n=args.max_images,
               resize=args.resize, skip_exists=args.skip_exists, prefix='train')

    get_images(validation, os.path.join(args.dataset, 'train'),
               max_workers=args.max_workers, n=args.max_images,
               resize=args.resize, skip_exists=args.skip_exists, prefix='valid')

    get_images(test, os.path.join(args.dataset, 'test'),
               max_workers=args.max_workers, n=args.max_images,
               resize=args.resize, skip_exists=args.skip_exists)


if __name__ == '__main__':
    main()
