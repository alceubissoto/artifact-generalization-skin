import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset

class SkinDataset(ConfounderDataset):
    """
    Skin dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(self, root_dir, root_csv, target_name, confounder_names,
                 model_type, augment_data):
        self.root_dir = root_dir
        self.root_csv = root_csv
        self.target_name = target_name
        self.confounder_names = ["dark_corner","hair","gel_border","gel_bubble","ink","ruler","patches"]
        self.augment_data = augment_data
        self.model_type = model_type
        print(confounder_names)
        # Read in attributes
        self.attrs_df = pd.read_csv('/artifact_based_generalization_isic22/isic_inferred_wocarcinoma.csv')
        self.data = pd.read_csv(self.root_csv)
        self.attrs_df = self.attrs_df[self.attrs_df['image'].isin(self.data['image'])]

        # Split out filenames and attribute names
        self.data_dir = self.root_dir
        self.filename_array = self.attrs_df['image'].values
        self.filename_array = np.array([name + '.jpeg' for name in self.filename_array])
        self.attrs_df = self.attrs_df.drop(labels='image', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()
        print(self.attr_names)

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        #self.attrs_df = self.attrs_df.values
        for cfd in self.confounder_names:
            self.attrs_df[cfd][self.attrs_df[cfd] > 0.6] = 1
            self.attrs_df[cfd][self.attrs_df[cfd] <= 0.6] = 0
        #self.attrs_df[self.attrs_df == -1] = 0

        self.attrs_df = self.attrs_df.values
        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        #print('target idx', target_idx)
        self.y_array = self.attrs_df[:, target_idx]
        print("y:", self.y_array)
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        print('confounder_idx', self.confounder_idx)
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        print('confounder_id', confounder_id)
        self.confounder_array = confounder_id

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Read in train/val/test splits
        #self.split_df = pd.read_csv(
        #    os.path.join(root_dir, 'data', 'list_eval_partition.csv'))
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        for k, v in self.split_dict.items():
            if k in self.root_csv:
                self.split_array = np.ones(len(self.y_array)) * v
        #if model_attributes[self.model_type]['feature_type']=='precomputed':
        #    self.features_mat = torch.from_numpy(np.load(
        #        os.path.join(root_dir, 'features', model_attributes[self.model_type]['feature_filename']))).float()
        #    self.train_transform = None
        #    self.eval_transform = None
        #else:
        #    self.features_mat = None
        self.train_transform = get_transform_skin(train=True)
        self.eval_transform = get_transform_skin(train=False)

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)


def get_transform_skin(train):
    orig_w = 299
    orig_h = 299
    target_resolution = (orig_w, orig_h)

    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform
