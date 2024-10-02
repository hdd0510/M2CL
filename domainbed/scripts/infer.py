# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This toy script is for PACS inference only. The hyperparameters are adjusted manually for PACS settings
"""

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import warnings
warnings.filterwarnings("ignore")


def infer_1_algorithm(args, algorithm, pretrain, image):
    global index2label
    
    ## Initialization
    algorithm_dict = None
    if pretrain is not None:
        algorithm_dict = torch.load(pretrain, map_location='cpu')
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if args.batch_size is not None:
        hparams['batch_size'] = args.batch_size
    if args.lr is not None:
        hparams['lr'] = args.lr
        
    hparams['lparam'] = args.lparam
    hparams['temp'] = args.temp

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    algorithm_class = algorithms.get_algorithm_class(algorithm)
    algorithm = algorithm_class((3, 224, 224, ), 7, 3, hparams) # for PACS only

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict['model_dict'])
        
        # print(f"Load checkpoint from {args.pretrain}")

    index2label = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    ## Inference ##
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(image).unsqueeze(0)
    
    algorithm.to(device)
    pred = misc.inference(algorithm, x, None, device)
    pred = torch.argmax(pred, dim=1).cpu().item()
    
    # print(f"Result [{args.algorithm}]: ", index2label[pred])
    return index2label[pred]
    
def infer_1_img_M_algos(infer_path, out_dir):
    global index2label
    
    image = Image.open(infer_path).convert("RGB")
    
    algorithms = ['ERM', 'M2CL', 'MMD']
    pretrains = [
        './ERM/model_best_env3_out_acc.pkl',
        './M2CL/model_best_env3_out_acc.pkl', 
        './MMD/model_best_env3_out_acc.pkl'
    ]
    
    preds = []
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(16, 5))
    for i, algorithm in enumerate(algorithms):
        pretrain = pretrains[i]
        pred = infer_1_algorithm(args, algorithm, pretrain, image)
        
        # saving inference image
        axs[i].imshow(np.asarray(image))
        axs[i].set_title(f"{algorithm} - {pred}")
    
    name = '_'.join(infer_path.split("/")[-2:])
    save_path = os.path.join(out_dir, name)
    plt.savefig(save_path)
    plt.cla()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--temp', type=float)
    parser.add_argument('--lparam', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--domain', type=int, default=3, help='index for domain respectively, -1 for all domains')
    
    ## additional ##
    parser.add_argument('--pretrain', type=str, help='Pretrain weights to load', default=None)
    parser.add_argument('--infer_path', type=str, help='Image path for inference', required=None)
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    domain_list = ['art_painting', 'cartoon', 'photo', 'sketch', 'all']
    infer_paths_3 = [
            "./domainbed/data/PACS/sketch/dog/5281.png",
            "./domainbed/data/PACS/sketch/giraffe/7385.png",
            "./domainbed/data/PACS/sketch/horse/8504.png",
            "./domainbed/data/PACS/sketch/elephant/n02503517_792-1.png",
            "./domainbed/data/PACS/sketch/person/12097.png",
            "./domainbed/data/PACS/sketch/guitar/7638.png"
        ]
    infer_paths_0 = [
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/art_painting/dog/pic_007.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/art_painting/elephant/pic_005.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/art_painting/giraffe/pic_008.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/art_painting/guitar/pic_007.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/art_painting/horse/pic_007.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/art_painting/house/pic_008.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/art_painting/person/pic_006.jpg'
        ]
    infer_paths_1 = [
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/cartoon/dog/pic_001.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/cartoon/elephant/pic_001.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/cartoon/giraffe/pic_001.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/cartoon/guitar/pic_001.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/cartoon/horse/pic_001.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/cartoon/house/pic_001.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/cartoon/person/pic_001.jpg'
        ]
    infer_paths_2 = [
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/photo/dog/056_0001.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/photo/elephant/064_0002.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/photo/giraffe/084_0004.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/photo/guitar/063_0005.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/photo/horse/105_0007.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/photo/house/pic_004.jpg',
            '/Users/dinzdzun/Downloads/m2cl/domainbed/data/PACS/photo/person/253_0003.jpg'
        ]
    domain = domain_list[args.domain]
    if domain == 'sketch':
        infer_paths =  infer_paths_3
    elif domain == 'art_painting':
        infer_paths = infer_paths_0
    elif domain == 'cartoon':
        infer_paths = infer_paths_1
    elif domain == 'photo':
        infer_paths = infer_paths_2
    else:
        infer_dir = [infer_paths_0, infer_paths_1, infer_paths_2, infer_paths_3]
    if domain == 'all':
        for i, domain in enumerate(domain_list[:-1]):
            infer_paths = infer_dir[i]
            out_dir = args.output_dir + f"/{domain}"
            os.makedirs(out_dir, exist_ok=True)
            
            for infer_path in infer_paths:
                infer_1_img_M_algos(infer_path, out_dir)
                print(f"Saved {os.path.basename(infer_path)} to: ", out_dir)
    else:
        out_dir = args.output_dir + f"/{domain}"
        os.makedirs(out_dir, exist_ok=True)
        
        for infer_path in infer_paths:
            infer_1_img_M_algos(infer_path, out_dir)
            print(f"Saved {os.path.basename(infer_path)} to: ", out_dir)