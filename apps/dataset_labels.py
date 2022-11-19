# the dataset.json structure is follow https://github.com/NVlabs/stylegan2-ada-pytorch/issues/18
import argparse
from torchvision import datasets
from PIL import Image
import os
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='/opt/data/private/customer/data/afhq/train/dataset.json')
    parser.add_argument('--source', type=str, default='/opt/data/private/customer/data/afhq/train/')
    parser.add_argument('--min_size', type=int, default=None, help='Minimum image size to not drop it')

    args = parser.parse_args()

    if args.min_size is None:
        valid_file = None
    else:
        valid_file = lambda x: min(Image.open(x).size) >= args.min_size

    imgset = datasets.ImageFolder(args.source, is_valid_file=valid_file)

    labels = {'labels': []}
    for img in imgset.imgs:
        fname, lable = img

        arch_fname = os.path.relpath(fname, args.source)
        arch_fname = arch_fname.replace('\\', '/')

        labels['labels'].append([arch_fname, lable])

    with open(args.out, 'w') as f:
        json.dump(labels, f)
