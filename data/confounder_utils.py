import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.multinli_dataset import MultiNLIDataset
from data.skin_dataset import SkinDataset
import pickle
################
### SETTINGS ###
################

confounder_settings = {
    'CelebA':{
        'constructor': CelebADataset
    },
    'CUB':{
        'constructor': CUBDataset
    },
    'MultiNLI':{
        'constructor': MultiNLIDataset
    },
    'Skin':{
        'constructor': SkinDataset
    },
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    if args.dataset == 'Skin':
        if train:
            splits = [args.train_csv, args.val_csv, args.test_csv]            
        else:
            splits = [args.test_csv]
            
        full_dataset = [confounder_settings[args.dataset]['constructor'](
            root_dir=args.root_dir,
            root_csv=split,
            target_name=args.target_name,
            confounder_names=args.confounder_names,
            model_type=args.model,
            augment_data=args.augment_data) for split in splits]
        
        dro_subsets = [DRODataset(dataset, process_item_fn=None, n_groups=dataset.n_groups,
                                  n_classes=dataset.n_classes, group_str_fn=dataset.group_str) \
                       for dataset in full_dataset]

    else:
        full_dataset = confounder_settings[args.dataset]['constructor'](
            root_dir=args.root_dir,
            target_name=args.target_name,
            confounder_names=args.confounder_names,
            model_type=args.model,
            augment_data=args.augment_data)
        if return_full_dataset:
            return DRODataset(
                full_dataset,
                process_item_fn=None,
                n_groups=full_dataset.n_groups,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str)

        if train:
            splits = ['train', 'val', 'test']
        else:
            splits = ['test']
        subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
        dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                                  n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                       for split in splits]
    return dro_subsets
