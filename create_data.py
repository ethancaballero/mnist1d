import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d

from data import make_dataset, get_dataset_args
from utils import set_seed, to_pickle

import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    filler
    ''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset_size', type=int, default=4e7, metavar='tds',
                        help='')
    parser.add_argument('--save_folder', type=str, default='/home/ethancab/projects/rrg-bengioy-ad/ethancab/data/mnist1d/mnist1d_seeds')
    flags = parser.parse_args()

args = get_dataset_args()

args.seed = flags.seed

args.train_split = 1.0

args.num_samples = int(flags.dataset_size)

set_seed(args.seed)
args.shuffle_seq = False

start_time = time.time()
data = make_dataset(args=args)  # make the dataset

print("time", time.time() - start_time)

print(data['x'].dtype)

#to_pickle(data, "seeds/mnist1d_data__seed_"+str(args.seed)+".pkl")

to_pickle(data, flags.save_folder+"/mnist1d_data__size_"+str(flags.dataset_size)+"__seed_"+str(args.seed)+".pkl")
