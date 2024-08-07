import os

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

ROOTDIR = r"GRAPE\CFPs_30crop"

def get_imgpath(rootdir, imgname):
    return os.path.join(rootdir, imgname)
                
class GlauProg_Dataset(Dataset):
    """
    Glaucoma progression prediction from baseline fundus image dataset.
    The image file paths and accompanying labels (G-RISK/year = G-RISK SLOPE) are stored in a single .csv file. 
    """

    def __init__(self, imglab_csv, transform=None, geteye=False, getbaselinegrisk=False):
        """
        Args:
            imglab_csv (string): Path to the csv file with included images and progression labels.
            rootdir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_labels = pd.read_csv(imglab_csv)
        self.transform = transform
        self.geteye = geteye
        self.getbaselinegrisk = getbaselinegrisk

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgname = self.img_labels.iloc[idx, self.img_labels.columns.get_loc('img')].split('/')[-1]
        imgname = get_imgpath(ROOTDIR, imgname)
        image = np.array(Image.open(imgname))

        slopeR0 = self.img_labels.iloc[idx, self.img_labels.columns.get_loc('grisk_slope_reg0')]
        label = max((slopeR0), 0)*100

        eye = self.img_labels.iloc[idx, self.img_labels.columns.get_loc('eye')]
        baseline_grisk = self.img_labels.iloc[idx, self.img_labels.columns.get_loc('baseline_grisk_reg0')]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if not self.geteye and not self.getbaselinegrisk:
            return image, label
        else:
            return image, label, eye, baseline_grisk
