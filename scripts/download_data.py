#!/usr/bin/env python3
"""Download public image datasets into the layout expected by cmc.data."""
import argparse
import os
from pathlib import Path
from torchvision import datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('CMC_DATA_ROOT', './data')))
    parser.add_argument('--datasets', nargs='+', choices=['mnist', 'fashion', 'cifar10', 'cifar100'], default=['mnist'])
    args = parser.parse_args()
    classes = dict(mnist=datasets.MNIST, fashion=datasets.FashionMNIST,
                   cifar10=datasets.CIFAR10, cifar100=datasets.CIFAR100)
    for name in args.datasets:
        directory = args.root / 'torchvision' if name.startswith('cifar') else args.root
        for train in (True, False):
            classes[name](root=str(directory), train=train, download=True)
        print(f'{name}: ready in {directory}')


if __name__ == '__main__':
    main()
