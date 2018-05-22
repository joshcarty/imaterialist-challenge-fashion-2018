import argparse
import json
import os

import pandas as pd

from PIL import Image


def load_json(path):
    with open(path, 'r') as f:
        labels = json.load(f)['annotations']
    return labels


def load_filenames(path, validate=True):
    files = os.listdir(path)

    if validate:
        files = validate_images(files, path)

    files = [file.split('.')[0] for file in files
             if file.endswith('.jpg')]

    return files


def validate_images(files, path):
    paths_files = [(os.path.join(path, file), file)
                   for file in files]

    valid_files = []
    for path, file in paths_files:
        try:
            Image.open(path)
            valid_files.append(file)
        except OSError:
            continue

    return valid_files


def create_dataframe(labels, dataset, prefix=None):
    df = pd.DataFrame(labels)

    if prefix:
        df['imageId'] = df['imageId'].apply(lambda f: '{}_{}'.format(prefix, f))

    # Ensure file is downloaded
    files = load_filenames(dataset)
    df = df[df['imageId'].isin(files)]


    df['labelId'] = df['labelId'].apply(" ".join)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_json', required=True)
    parser.add_argument('--valid_json', required=True)
    parser.add_argument('--outpath', required=True)
    parser.add_argument('--validate', default=True, type=bool)
    return parser.parse_args()


def main():
    args = parse_args()

    train_labels = load_json(args.train_json)
    valid_labels = load_json(args.valid_json)

    df_train = create_dataframe(train_labels, args.dataset, prefix='train')
    df_valid = create_dataframe(valid_labels, args.dataset, prefix='valid')

    df = pd.concat([df_train, df_valid])

    df.to_csv(args.outpath, index=False)


if __name__ == '__main__':
    main()
