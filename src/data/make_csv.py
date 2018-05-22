import argparse
import json
import os

import pandas as pd


def load_json(path):
    with open(path, 'r') as f:
        labels = json.load(f)['annotations']
    return labels


def load_filenames(path):
    files = os.listdir(path)
    files = [file.split('.')[0]
             for file in files
             if file.endswith('.jpg')]
    return files


def create_dataframe(labels, dataset):
    df = pd.DataFrame(labels)
    files = load_filenames(dataset)
    df = df[df['imageId'].isin(files)]
    df['labelId'] = df['labelId'].apply(" ".join)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--inpath', required=True)
    parser.add_argument('--outpath', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = load_json(args.inpath)
    df = create_dataframe(labels, args.dataset)
    df.to_csv(args.outpath, index=False)


if __name__ == '__main__':
    main()
