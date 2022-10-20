import os
import os.path
import torch
import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# TODO: Make target_field optional for unannotated datasets.
class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                 loader=default_loader, transform=None,
                 target_transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 split=None, environment=None, onlylabels=None, 
                 subset=None):
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform
        self.target_transform = target_transform
        self.add_extension = add_extension
        self.environment = environment
        self.onlylabels = onlylabels
        self.subset = subset
        self.data = pd.read_csv(csv_file)
 
        # Environments
        if self.environment is not None:
            all_envs = ["dark_corner","hair","gel_border","gel_bubble","ruler","ink","patches"]
            for env in all_envs:
                if env in self.environment:
                    self.data = self.data[self.data[env] >= 0.6]
                else:
                    self.data = self.data[self.data[env] <= 0.6]
            self.data = self.data.reset_index()


        # Only Label
        if self.onlylabels is not None:
            self.onlylabels = [int(i) for i in self.onlylabels]
            self.data = self.data[self.data[self.target_field].isin(self.onlylabels)]
            self.data = self.data.reset_index()
   
        # Subset
        if self.subset is not None:
            self.data = self.data[self.data['image'].isin(self.subset)]
            self.data = self.data.reset_index()
 
        # Split
        if split is not None:
            with open(split, 'r') as f:
                selected_images = f.read().splitlines()
            self.data = self.data[self.data[image_field].isin(selected_images)]
            self.data = self.data.reset_index()

        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(self.data['label'].value_counts())
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls]
                                for cls in self.data['label']]
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]

        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()

        if type(limit) == int:
            limit = (0, limit)
        if type(limit) == tuple:
            self.data = self.data[limit[0]:limit[1]]
            self.data = self.data.reset_index()

        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        print('Found {} images from {} classes.'.format(len(self.data),
                                                        len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.data[self.target_field].value_counts())
            print("    Class '{}' ({}): {} images.".format(
                class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension:
            path = path + self.add_extension
        sample = self.loader(path)
        target = self.class_to_idx[self.data.loc[index, self.target_field]]
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except:
                sample = np.array(sample)
                sample = self.transform(image=sample.astype(np.uint8))["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, target

    def __len__(self):
        return len(self.data)


class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        return super().__getitem__(i), name
    
class CSVDatasetWithMask(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        mask = Image.open('/deconstructing-bias-skin-lesion/isic2019-seg-299/{}.png'.format(name))
        mask = get_transform_mask(mask).cpu().data.numpy()
        mask = np.squeeze(mask)
        mask[mask>0.1] = 1.0
        mask[mask<=0.1] = 0.0
        return super().__getitem__(i), mask
    
def get_transform_mask(mask):
    
    transform_mask = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.0], [1.0])
    ])
    
    return transform_mask(mask)
    