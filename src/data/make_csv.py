import argparse
import json

import pandas as pd


def load_json(path):
    with open(path, 'r') as f:
        labels = json.load(f)['annotations']
    return labels


def create_dataframe(labels):
    df = pd.DataFrame(labels)
    df['labelId'] = df['labelId'].apply(" ".join)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True)
    parser.add_argument('--outpath', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = load_json(args.inpath)
    df = create_dataframe(labels)
    df.to_csv(args.outpath, index=False)


if __name__ == '__main__':
    main()
