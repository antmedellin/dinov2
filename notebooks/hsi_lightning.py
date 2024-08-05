# main.py
# ! pip install torchvision
import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L
import os
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.tuner import Tuner
import torchvision.transforms as transforms

from torch.utils.data import Dataset
import albumentations as A 
from os.path import join
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
# from torchmetrics import IoU, Accuracy, Precision, Recall
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics.segmentation import MeanIoU
# from torchmetrics.functional.segmentation import mean_iou
import torch.optim.lr_scheduler as lr_scheduler 
from lightning.pytorch.callbacks import ModelCheckpoint
import cv2
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for GUI-based environments
import matplotlib.pyplot as plt  
from osgeo import gdal
import json
import seaborn as sns
import torchvision 
import sys
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation, MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from PIL import Image
from transformers import BatchFeature
# from datasets import Dataset
import yaml

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
# pip install segmentation-models-pytorch

# pip install lightning tensorboard torch-tb-profiler pandas matplotlib seaborn
# pip install --upgrade torchmetrics
# sudo apt-get install python3-tk -y

# tensorboard --logdir=./lightning_logs/
# ctrl shft p -> Python: Launch Tensorboard  select lightning logs
  

def extract_rgb(cube, red_layer=70 , green_layer=53, blue_layer=19):

    red_img = cube[ red_layer,:,:]
    green_img = cube[ green_layer,:,:]
    blue_img = cube[ blue_layer,:,:]
        
    data=np.stack([red_img,green_img,blue_img], axis=-1)
    
    # convert from x,y,channels to channels, x, y
    # data = np.transpose(data, (2, 0, 1))
    
    return data 
  
def GDAL_imreadmulti(file_name):
    # Open the dataset
    dataset = gdal.Open(file_name)

    # Check if opened
    if dataset:
      # print("Dataset opened...")
      width = dataset.RasterXSize
      height = dataset.RasterYSize
      num_bands = dataset.RasterCount

      image_bands = []

      for band_num in range(1, num_bands+1):
        band = dataset.GetRasterBand(band_num)

        # Read band data
        band_data = band.ReadAsArray()

        # Create an OpenCV Mat from the band data
        band_mat = np.array(band_data, dtype='float32')

        # Correct the orientation of the image
        band_mat = np.transpose(band_mat)
        band_mat = cv2.flip(band_mat, 1)
        
        # Normalize the band data to the range [0, 1]
        band_min = band_mat.min()
        band_max = band_mat.max()
        normalized_band_mat = (band_mat - band_min) / (band_max - band_min)
        band_mat = normalized_band_mat

        # Apply threshold and convert to 8-bit unsigned integers
        _, band_mat = cv2.threshold(band_mat, 1.0, 1.0, cv2.THRESH_TRUNC)
        band_mat = cv2.convertScaleAbs(band_mat, alpha=(255.0))

        # Add the processed band to the list
        image_bands.append(band_mat)
        
        cube=np.array(image_bands)
        
        # convert from x,y,channels to channels, x, y
        # data = np.transpose(cube, (2, 0, 1))
        
        

      return True, cube

    else:
      print("GDAL Error: ", gdal.GetLastErrorMsg())
      return False, []

def read_hsd(filename: str):
    # def load_hsi(file_path: str):
    data_dict = torch.load(filename)
    height = data_dict["height"]
    width = data_dict["width"]
    SR = data_dict["SR"]
    average = data_dict["average"]
    coeff = data_dict["coeff"]
    scoredata = data_dict["scoredata"]

    temp = torch.mm(scoredata, coeff)
    data = (temp + average).reshape((height, width, SR))

    # return data.numpy()
    print(data.shape, height, width, SR)
    
    
    return data

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        num = targets.size(0)
        logits = logits.view(num, -1)
        targets = targets.view(num, -1)
        intersection = (logits * targets).sum()
        dice = (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.dice_loss = DiceLoss()
        self.JaccardLoss = JaccardLoss(mode="multiclass")
        

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy_loss(logits, targets)
        jaccard_loss = self.JaccardLoss(logits, targets)
        return ce_loss + jaccard_loss # assumes equal weighting of both losses

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
        
        
class LIBHSIDataset(Dataset):
    def __init__(self, image_set,  root_dir, id2color, processor=None ,transform=None):
        
        # image_set # train ,test, validation
        self.transform = transform
        self.root = join(root_dir,image_set)
        self.processor = processor
        
        
        # Convert id2color to a numpy array for easier comparison
        self.id2color_np = np.array(list(id2color.values()))

        self.img_dir =  join(self.root, "reflectance_cubes")
        self.label_dir = join(self.root, "labels")
        
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.' + 'dat')]
        self.num_images = len( self.img_names  ) 

        assert self.num_images == len(os.listdir(self.label_dir))
        
        self.img_labels = [f for f in os.listdir(self.label_dir)]
        
        # sort img_names and img_labels
        self.img_names.sort()
        self.img_labels.sort()

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        label_name, ext_label = os.path.splitext(self.img_labels[idx])
        
        hsi_name, ext_hsi = os.path.splitext(self.img_names[idx])
        
        assert label_name == hsi_name # make sure they have the same name 
        
        # read the label image 
        label_path = join(self.label_dir, self.img_labels[idx])
        label_img = Image.open(label_path).convert('RGB')
        
        label_img_np = np.array(label_img) # uint8 x,y,channels
        
        
        # print(label_name)
        
        # label_img_np = label_img_np.astype(np.float32)
        # print(label_img_np.dtype, label_img_np.shape)
        
        #convert labeled rgb image to greyscale
        label_img_greyscale = np.zeros(label_img_np.shape[:2], dtype=np.uint8)
        for i, color in enumerate(self.id2color_np):
            # Find where in the target the current color is
            mask = np.all(label_img_np == color, axis=-1)
            
            # Wherever the color is found, set the corresponding index in target_new to the current class label
            label_img_greyscale[mask] = i
        
        hsi_path = join(self.img_dir, self.img_names[idx])
        _, hsi_img = GDAL_imreadmulti(hsi_path)

        
        rgb_img = extract_rgb(hsi_img) 

        hsi_img = np.transpose(hsi_img, (1, 2, 0)) # transpose to x,y,channels for albumnetations
        
        # apply transformations  # must be in x,y,channels format        
        if self.transform:
            hsi_img_transformed = self.transform(image=hsi_img)
            transformed = self.transform(image = rgb_img, mask = label_img_greyscale)
            label_img_greyscale = self.transform(image = label_img_greyscale)
        
            hsi_img, rgb_img, label_img_greyscale = torch.tensor(hsi_img_transformed['image']), torch.tensor(transformed['image']), torch.tensor(transformed['mask'])
        else:
            hsi_img, rgb_img, label_img_greyscale = torch.tensor(hsi_img), torch.tensor(rgb_img), torch.tensor(label_img_greyscale)
            
            
        #convert from x,y,channels to channels, x, y
        hsi_img = hsi_img.permute(2,0,1)
        rgb_img = rgb_img.permute(2,0,1)
        
        #convert from uint8 to float32
        hsi_img = hsi_img.float()
        rgb_img = rgb_img.float()
        
        # if self.processor:
            # rgb_img = self.processor(images = rgb_img, segmentation_maps = label_img_greyscale, task_inputs=["semantic"], return_tensors="pt")
            
            # rgb_img = {k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in rgb_img.items()}

            
        return hsi_img, rgb_img, label_img_greyscale

class HyperSpectralCityV2Dataset(Dataset):
    def __init__(self, image_set,  root_dir, transform=None):
        
        # image_set # train ,test, validation
        self.transform = transform
        if image_set == "train":
            # self.root = join(root_dir,"training dataset") V2_fixed_train_label
            self.root = join(root_dir,"V2_fixed_train_label")
        elif image_set == "test":
            self.root = join(root_dir,"testing_dataset")
        elif image_set == "validation":
            # self.root = join(root_dir,"validation dataset/testing dataset")
            self.root = join(root_dir,"V2_fixed_test_label")
        else:
            raise ValueError("image_set must be one of 'train', 'test', or 'validation'")
        
        # self.img_dir =  join(self.root, "hsi")
        self.img_dir =  join(self.root, "img") #hsi
        self.label_dir = join(self.root, "gt")
        
        # self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.' + 'pt')]
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.' + 'jpg')]
        self.num_images = len( self.img_names  ) 
        
        self.img_names.sort()
        
        # make sure the img_names are also in label_dir and create label_names
        # Ensure the img_names are also in label_dir and create label_names
        self.label_names = []
        for img_name in self.img_names:
            # label_name = 'rgb' + img_name.replace('.pt', '_gray.png')
            label_name =  img_name.replace('.jpg', '_gray.png')
            if label_name in os.listdir(self.label_dir):
                self.label_names.append(label_name)
                # print(label_name)   
            else:
                raise FileNotFoundError(f"Label file {label_name} not found in {self.label_dir}")
        
        # Ensure the number of images matches the number of labels
        assert self.num_images == len(self.label_names), "Mismatch between number of images and labels"
        
        
        # print(self.num_images)
        
        # assert self.num_images == 2*len(os.listdir(self.label_dir))
        
        # print(self.root)
        # self.root =  "/workspaces/hyper_city/training dataset"
        # print(self.root)
        # entries = os.listdir(self.root)
        # # print(entries)
        # directories = [entry for entry in entries if os.path.isdir(os.path.join(self.root, entry))]
        # # print(directories)
        # self.num_images = len( directories  ) 

        # self.directories = directories
        # id to label mapping for v2
        # 0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
        
        # v1 
        # 0: 'background', 1: 'car', 2: 'human', 3: 'road', 4: 'traffic light', 5: 'traffic sign', 6: 'tree', 7: 'building', 8: 'sky', 9: 'object
        

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        
        
        # print(self.img_names[idx], self.label_names[idx], self.img_dir, self.label_dir)
        
        
        label_name = os.path.join(self.label_dir, self.label_names[idx])
        hsi_name = os.path.join(self.img_dir, self.img_names[idx])
        
        label_img = Image.open(label_name)
        label_img = np.array(label_img)
        
        # if value equals 255 in label img then it is background so set to  19
        label_img[label_img == 255] = 19
        
        
 
        rgb_img = Image.open(hsi_name)
        rgb_img = np.array(rgb_img)
        
        hsi_img = rgb_img
        
        # print(f"Label image shape: {label_img.shape}")
        # print(label_img.dtype, label_img)
        
        # Load the hyperspectral image using torch
        # try:
        #     hsi_img = torch.load(hsi_name)
        #     print(f"Hyperspectral image shape: {hsi_img.shape}")
        # except RuntimeError as e:
        #     print(f"Error loading {hsi_name}: {e}")
        #     return None, None
        
        if self.transform:
            # hsi_img_transformed = self.transform(image=hsi_img)
            transformed = self.transform(image = rgb_img, mask = label_img, hsi_image = hsi_img)
            # label_img_greyscale = self.transform(image = label_img)
        
            hsi_img, rgb_img, label_img = torch.tensor(transformed['hsi_image']), torch.tensor(transformed['image']), torch.tensor(transformed['mask'])
        else:
            hsi_img, rgb_img, label_img = torch.tensor(hsi_img), torch.tensor(rgb_img), torch.tensor(label_img)
        # print(transformed, self.transform)   
        #convert from x,y,channels to channels, x, y
        hsi_img = hsi_img.permute(2,0,1)
        rgb_img = rgb_img.permute(2,0,1)
        
        #convert from uint8 to float32
        hsi_img = hsi_img.float()
        rgb_img = rgb_img.float()
        
        return hsi_img, rgb_img, label_img
     
        # print(os.listdir(os.path.join(self.root, self.directories[idx])))
        hsi_temp = self.directories[idx][3:]
        hsi_name = os.path.join(self.root, self.directories[idx],  hsi_temp[:-5] + ".hsd")
        
        label_name = os.path.join(self.root, self.directories[idx],"label_gray.png")
        # print(hsi_name, label_name)
        # make sure both of these files exist 
        if not os.path.exists(hsi_name) :
            print("File not found",  self.directories[idx])
            hsi_name = os.path.join(self.root, self.directories[idx],  hsi_temp[:-5] + "(1).hsd")
        if not os.path.exists(label_name):
            print("File not found",  self.directories[idx])
        
        assert os.path.exists(hsi_name) 
        assert os.path.exists(label_name)
        
        # read the label image 
        label_img = Image.open(label_name)
        
        label_img_greyscale = np.array(label_img) # uint8 x,y,channels
        # print(label_img_greyscale.shape, label_img_greyscale.dtype)
                
        
        # print unique color values in mask 
        # print(label_name, np.unique(label_img_greyscale))
        
        
        hsi_img = read_hsd(hsi_name)

        
        rgb_img = extract_rgb(hsi_img) 

        hsi_img = np.transpose(hsi_img, (1, 2, 0)) # transpose to x,y,channels for albumnetations
        
        # apply transformations  # must be in x,y,channels format 
        
        # print(hsi_img.shape, rgb_img.shape, label_img_greyscale.shape)
               
        if self.transform:
            hsi_img_transformed = self.transform(image=hsi_img)
            transformed = self.transform(image = rgb_img, mask = label_img_greyscale)
            label_img_greyscale = self.transform(image = label_img_greyscale)
        
            hsi_img, rgb_img, label_img_greyscale = torch.tensor(hsi_img_transformed['image']), torch.tensor(transformed['image']), torch.tensor(transformed['mask'])
        else:
            hsi_img, rgb_img, label_img_greyscale = torch.tensor(hsi_img), torch.tensor(rgb_img), torch.tensor(label_img_greyscale)
            

        #convert from x,y,channels to channels, x, y
        hsi_img = hsi_img.permute(2,0,1)
        rgb_img = rgb_img.permute(2,0,1)
        
        #convert from uint8 to float32
        hsi_img = hsi_img.float()
        rgb_img = rgb_img.float()
        
        # print(hsi_img.shape, rgb_img.shape, label_img_greyscale.shape)
        
        return hsi_img, rgb_img, label_img_greyscale
 
class UNET_SemanticSegmentation(L.LightningModule):
        def __init__(self, num_classes,repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=True , tokenW=64, tokenH=64, learning_rate = 1e-3, ignore_index=0 ,num_channels=204, num_workers=4, dataset_name=None, train_dataset=None, val_dataset=None, test_dataset = None, batch_size=2):
            super().__init__()
            
            
            self.save_hyperparameters()
            
            # load the dinov2 model 
            if half_precision:
                self.dinov2 = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
            else:
                self.dinov2= torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
                
            last_layer_params = list(self.dinov2.parameters())[-1]
            patch_descriptor_size = last_layer_params.shape[0]
            
            
            # Freeze the DINOv2 model. This allows for faster training. 
            for _, param in self.dinov2.named_parameters():
                param.requires_grad = False
            
            self.classifier = torch.nn.Conv2d(patch_descriptor_size, num_classes, (1,1))
           
            
            self.patch_descriptor_size = patch_descriptor_size
            self.tokenW = tokenW
            self.tokenH = tokenH
            self.learning_rate = learning_rate
            self.ignore_index = ignore_index
            
            self.num_workers = num_workers
            self.dataset_name = dataset_name
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
            # optimal batch size is found  so it is not set to a specific value
           
            
            self.tokenW = tokenW
            self.tokenH = tokenH
            # self.learning_rate = learning_rate
            self.ignore_index = ignore_index
            
            
            print("num classes:", num_classes)
            # performance metrics
            # self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            # self.loss_fn = FocalLoss(ignore_index=ignore_index)
            # self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction="mean")
            # self.loss_fn = JaccardLoss(mode="multiclass")
            self.loss_fn = CombinedLoss(ignore_index=ignore_index)

            
            self.train_miou = MeanIoU(num_classes=num_classes, per_class=False)
            self.test_miou = MeanIoU(num_classes=num_classes, per_class=False)
            self.val_miou = MeanIoU(num_classes=num_classes, per_class=False)
            
            self.train_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true", ignore_index=ignore_index)
            self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true", ignore_index=ignore_index)
            self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true", ignore_index=ignore_index)
            
            #  Calculate statistics for each label and average them
            self.train_acc_mean = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=ignore_index)
            self.val_acc_mean = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=ignore_index)
            self.test_acc_mean = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=ignore_index)
            
            #  Sum statistics over all labels
            self.train_acc_overall = MulticlassAccuracy(num_classes=num_classes, average="micro", ignore_index=ignore_index)
            self.val_acc_overall = MulticlassAccuracy(num_classes=num_classes, average="micro", ignore_index=ignore_index)
            self.test_acc_overall = MulticlassAccuracy(num_classes=num_classes, average="micro", ignore_index=ignore_index)
            
            
            # UNET 
            
            # reference 
            # https://segmentation-modelspytorch.readthedocs.io/en/latest/
            
            self.hsi_unet = smp.Unet( 'resnet152', in_channels=num_channels, classes=num_classes, encoder_depth=5)
            
            
            #reference
            # https://github.com/hamdaan19/UNet-Multiclass/blob/main/scripts/model.py
            
            # # need to modify for dual input
            # self.layers = [num_channels, 64, 128, 256, 512, 1024] # use same architecute for 3/200 bands (if higher then it should help to remove some of the noise and take the most information from the bands)
            
            # self.double_conv_downs = nn.ModuleList(
            # [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
            
            # self.up_trans = nn.ModuleList(
            #     [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
            #     for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            # self.double_conv_ups = nn.ModuleList(
            #     [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
            # self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
            # self.final_conv = nn.Conv2d(self.layers[1], num_classes, kernel_size=1)
            
            
            self.fusion_classifier = torch.nn.Sequential(
                nn.ConvTranspose2d(num_classes*2, int(num_classes*2*1.1), kernel_size=7, stride=2), # upsample kernel size 7  since dinov2 has a patch descriptor size of 14x14
                nn.ReLU(),
                nn.Conv2d(int(num_classes*2*1.1), num_classes, kernel_size=7, stride=2)
            )
            
            
    
        def forward(self, hsi_pixel_values, rgb_pixel_values):
           

            # test out unet without rgb right now so just hyperspectral 
            # concat_layers = []
            # x = hsi_pixel_values
            
            # # down layers 
            # for down in self.double_conv_downs:
            #     # x_new = self.adjust_channels(hsi_pixel_values) 
            #     # print(1)
            #     x = down(x)
            #     # print(2)
            #     if down != self.double_conv_downs[-1]:
            #         concat_layers.append(x)
            #         x = self.max_pool_2x2(x)
        
            # concat_layers = concat_layers[::-1]
            
            # # up layers 
            # for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            #     x = up_trans(x)
            #     if x.shape != concat_layer.shape:
            #         x = torchvision.transforms.functional.resize(x, concat_layer.shape[2:])
                
            #     concatenated = torch.cat((concat_layer, x), dim=1)
            #     x = double_conv_up(concatenated)
                
            # x = self.final_conv(x)
            
            x = self.hsi_unet(hsi_pixel_values)
            
            # dino part 
            if torch.isnan(rgb_pixel_values).any():
                rgb_pixel_values = torch.nan_to_num(rgb_pixel_values, nan=0.0)
            assert not torch.isnan(rgb_pixel_values).any(), "NaN values in input pixel_values"            
            embeddings = self.dinov2.get_intermediate_layers(rgb_pixel_values)[0].squeeze()
            
            assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            embeddings = embeddings.reshape(-1, self.tokenW, self.tokenH, self.patch_descriptor_size)
            embeddings = embeddings.permute(0,3,1,2)
        
            # assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            logits = self.classifier(embeddings)
            # print( logits[0])
            assert not torch.isnan(logits).any(), "NaN values in logits"
            y = torch.nn.functional.interpolate(logits, size=rgb_pixel_values.shape[2:], mode="bilinear", align_corners=False)
            
            
            # # fuse logits from x and y 
            combined_embeddings = torch.cat([x, y], dim=1) # concatenate along the channel dimension so descriptors are combined
            # print(combined_embeddings.shape)
            z = self.fusion_classifier(combined_embeddings)
            # print(z.shape)
            return x, y, z

      
        
        def __double_conv(self, in_channels, out_channels):
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return conv
        
        def log_cf(self, result_cf, step_type):
            
            confusion_matrix_computed = result_cf.detach().cpu().numpy()
            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize = (50,45))
            fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
            plt.close(fig_)
            self.loggers[0].experiment.add_figure(f"Confusion Matrix {step_type}", fig_, self.current_epoch)
            
        def log_data(self, step_type, logits, labels, loss):
            
            preds = torch.argmax(logits, dim=1)
            
            # Check the shapes of preds and labels
            # print(f"Shape of preds: {preds.shape}, dtype: {preds.dtype}")
            # print(f"Shape of labels: {labels.shape}, dtype: {labels.dtype}")
            
            assert preds.shape == labels.shape, "Predictions and labels must have the same shape"
            # Check for NaNs or Infs
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                raise ValueError("preds contain NaNs or Infs")
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                raise ValueError("labels contain NaNs or Infs")
            
            # Check unique values
            # print(f"Unique values in preds: {torch.unique(preds)}")
            # print(f"Unique values in labels: {torch.unique(labels)}")

            # Check number of classes
            # num_classes_preds = len(torch.unique(preds))
            # num_classes_labels = len(torch.unique(labels))
            # print(f"Number of classes in preds: {num_classes_preds}")
            # print(f"Number of classes in labels: {num_classes_labels}")
            
            # Ensure preds has the correct number of classes
            # if num_classes_preds != self.train_miou.num_classes:
            #     raise ValueError(f"Number of classes in preds ({num_classes_preds}) does not match expected ({self.train_miou.num_classes})")

    
            
            if step_type == "train":
                # result_cf = self.train_confusion_matrix(preds, labels) # not used in training loop
                result_miou = self.train_miou(preds, labels)
                result_acc_overall = self.train_acc_overall(preds, labels)
                results_acc_mean = self.train_acc_mean(preds, labels)
                # print("train", result_miou, result_acc_overall, results_acc_mean)
            elif step_type == "val":
                # result_cf = self.val_confusion_matrix(preds, labels)
                result_miou = self.val_miou(preds, labels)
                result_acc_overall = self.val_acc_overall(preds, labels)
                results_acc_mean = self.val_acc_mean(preds, labels)
                # self.log_cf(result_cf, step_type)
            elif step_type == "test":
                result_cf = self.test_confusion_matrix(preds, labels)
                result_miou = self.test_miou(preds, labels)
                result_acc_overall = self.test_acc_overall(preds, labels)
                results_acc_mean = self.test_acc_mean(preds, labels)
                self.log_cf(result_cf, step_type)
            else:
                raise ValueError("step_type must be one of 'train', 'val', or 'test'")
            
            self.log(f"{step_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_accuracy_overall", result_acc_overall, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_accuracy_mean", results_acc_mean, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_miou", result_miou, on_step=False, on_epoch=True, prog_bar=True)


        
        def training_step(self, batch, batch_idx):
            
            step_type = "train"
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits_hsi, logits_rgb, logits_fused = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss_hsi = self.loss_fn(logits_hsi, labels) 
            loss_rgb = self.loss_fn(logits_rgb, labels)
            loss_fused = self.loss_fn(logits_fused, labels)
       
            combined_loss = loss_hsi + loss_rgb + 4*loss_fused # penalize fused loss a lot more since it is final output
            
            
            self.log_data(step_type, logits_fused, labels, combined_loss)

            
            return combined_loss
        
        def test_step(self, batch, batch_idx):
            
            step_type = "test"
            
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits_hsi, logits_rgb, logits_fused  = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss_hsi = self.loss_fn(logits_hsi, labels) 
            loss_rgb = self.loss_fn(logits_rgb, labels)
            loss_fused = self.loss_fn(logits_fused, labels)
       
            combined_loss = loss_hsi + loss_rgb + 4*loss_fused
            
            
            self.log_data(step_type, logits_fused, labels, combined_loss)

            
            return combined_loss
        
        def validation_step(self, batch, batch_idx):
            
            step_type = "val"
            
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits_hsi, logits_rgb, logits_fused = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss_hsi = self.loss_fn(logits_hsi, labels) 
            loss_rgb = self.loss_fn(logits_rgb, labels)
            loss_fused = self.loss_fn(logits_fused, labels)
       
            combined_loss = loss_hsi + loss_rgb + 4*loss_fused
            
            
            self.log_data(step_type, logits_fused, labels, combined_loss)

            
            return combined_loss
        
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
            # return optimizer
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
            return {
            'optimizer': optimizer,
            "lr_scheduler": scheduler
             }       

        def train_dataloader(self):
            
            
            return  DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn,num_workers=num_workers)
        
        def val_dataloader(self):
            
            return  DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=num_workers)
        def test_dataloader(self):
            
            return  DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=num_workers)

def collate_fn(inputs):

    batch = dict()
    batch["hsi_pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["rgb_pixel_values"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[2] for i in inputs], dim=0).long()

    return batch

def collate_fn_oneformer(inputs):

    one_batch = list(zip(*inputs))
 
    batch = dict()
    batch["hsi_pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[2] for i in inputs], dim=0).long()
    batch["rgb_pixel_values"] = torch.stack([i[1] for i in inputs], dim=0)
    
    # task_inputs = ["semantic"] * len(one_batch[1])
    
    # one_batch_new = processor(images = one_batch[1], segmentation_maps = one_batch[2], return_tensors="pt") # task_inputs=task_inputs, 
    
    # batch["rgb_pixel_values"] = one_batch_new
    # # batch["rgb_pixel_values"] = {k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in one_batch_new.items()}

    # one_batch_new["original_images"] = one_batch[1]
    # one_batch_new["original_segmentation_maps"] = one_batch[2]

    # batch["rgb_pixel_values"] = {k: v for k, v in one_batch_new.items() if isinstance(v, torch.Tensor)}

    

    return batch

dataset = 'LIB-HSI'
# dataset = 'HyperCity' # using v1 

if dataset == 'LIB-HSI':
    dataset_dir='/workspaces/LIB-HSI'
    rgb_data_json = '/workspaces/dinov2/notebooks/lib_hsi_rgb.json'
elif dataset == 'HyperCity':
    dataset_dir='/workspaces/hyper_city2'
else:
    raise ValueError("Dataset must be one of 'LIB-HSI' or 'HyperCity'")

batch_size = 2
ignore_index=-1
num_workers = 3 #  os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() is None
initial_lr =  0.0001 #0.00001 
swa_lr = 0.01
# these should be multiple of 14 for dino model 
img_height = 448
img_width = 448
max_num_epochs = 100
accumulate_grad_batches = 5# 5 # increases the effective batch size  # 1 means no accumulation # more important when batch size is small or not doing multi gpu training
oneformer= False
grad_clip_val = 5 # clip gradients that have norm bigger than this
training_model = True
tuning_model = False
min_epochs = 20


if oneformer:
    # processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    processor = MaskFormerImageProcessor(ignore_index=ignore_index, do_reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False) # used also in collate_fn 

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    total_vram = gpu_properties.total_memory / 1e9  # Convert bytes to GB
    print(f"Total VRAM on device: {total_vram:.2f} GB")
else:
    print("CUDA is not available. Check if GPU is available or if PyTorch is installed with CUDA.")


# Define mean and standard deviation for normalization
# Use the same value for all channels
num_channels = 204
mean = [0.45] * num_channels  # Assuming num_channels is defined elsewhere
std = [0.225] * num_channels   # Assuming num_channels is defined elsewhere



test_transform = A.Compose([
    A.Resize(width=img_width, height=img_height), # dinov2 has a patch descriptor size for 14x14 pixels, so we need to resize the image to a multiple of 14. This will also affect the tokens. divide dimensions by 14 and set to dimensions of tokens, larger resolutions will lead to better performance
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0)

], additional_targets={"hsi_image": "image"})

# https://medium.com/pytorch/multi-target-in-albumentations-16a777e9006e
# noramalize 0.45 / 0.225 mean std




train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomCrop(width=450, height=450),
    
    A.Resize(width=img_width, height=img_height), 
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0)

], additional_targets={"hsi_image": "image"})

if dataset == 'LIB-HSI':
    file_data =  open(rgb_data_json)
    file_contents = json.load(file_data)
    ignore_index=0 # background class, if this is not set then the model will throw an error during training

    id2label ={}
    id2color = {}
    for i, item in enumerate(file_contents['items'], start=0):
        id2label[i] = item['name']
        id2color[i] = [item['red_value'], item['green_value'], item['blue_value']]
        
    print(id2label)
    # print(id2color)
    num_classes = len(id2label)
    # print("num classes",num_classes)
    num_channels = 204 
    if oneformer:
        pass
    else:
        train_dataset = LIBHSIDataset(image_set="train", root_dir=dataset_dir, id2color=id2color, transform=train_transform)
        test_dataset = LIBHSIDataset(image_set="test", root_dir=dataset_dir, id2color=id2color,  transform=test_transform)
        val_dataset = LIBHSIDataset(image_set="validation", root_dir=dataset_dir, id2color=id2color, transform=test_transform)



elif dataset == "HyperCity":
    num_classes = 20#10 # 19 class + ignore index
    ignore_index = 19 #0
    num_channels = 3#129
    train_dataset = HyperSpectralCityV2Dataset(image_set="train", root_dir=dataset_dir,  transform=train_transform)
    test_dataset = HyperSpectralCityV2Dataset(image_set="validation", root_dir=dataset_dir,  transform=test_transform)
    val_dataset = HyperSpectralCityV2Dataset(image_set="validation", root_dir=dataset_dir,  transform=test_transform)
  
  
# train_dataset[0]
# print(train_dataset[0])

# plt.imshow(train_dataset[0][1])
# plt.figure()
# plt.imshow(train_dataset[0][2])
# plt.figure()
# plt.imshow(train_dataset[0][0])

# print(train_dataset[0][0].shape, train_dataset[0][1].shape, train_dataset[0][2].shape)

# plt.show()

# print(train_dataset[0][1], train_dataset[0][2], train_dataset[0][0])

# sys.exit()

# model = DinoV2SemanticSegmentation(num_classes=num_classes, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=False, tokenW=img_width//14, tokenH=img_height//14, learning_rate=initial_lr, ignore_index=ignore_index)

model = UNET_SemanticSegmentation(num_classes=num_classes, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=False, tokenW=img_width//14, tokenH=img_height//14, learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers, dataset_name = dataset, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

# model = OneFormer_SemanticSegmentation(num_classes=num_classes, learning_rate=initial_lr, ignore_index=ignore_index, processor=processor, num_channels= num_channels)

if oneformer:
        
        train_dataset = LIBHSIDataset(image_set="train", root_dir=dataset_dir, id2color=id2color, transform=test_transform, processor=processor)
        test_dataset = LIBHSIDataset(image_set="test", root_dir=dataset_dir, id2color=id2color,  transform=test_transform, processor=processor)
        val_dataset = LIBHSIDataset(image_set="validation", root_dir=dataset_dir, id2color=id2color, transform=test_transform, processor=processor)


print("train dataset length", train_dataset.__len__())
print("test dataset length", test_dataset.__len__())
print("val dataset length", val_dataset.__len__())

# train_dataset[0]
# print(train_dataset[0])
# sys.exit()

if oneformer:

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_oneformer,num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_oneformer,num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_oneformer,num_workers=num_workers)
else: 
    
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=num_workers)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=num_workers)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=num_workers)
    # dataloaders implemented in model    
    pass


# train_dataset[0]
# batch = next(iter(train_dataloader))
# for k,v in batch.items():
#   print(k)
#   if isinstance(v, torch.Tensor):
#     print(k,v.shape)
#   else:
#       for l,m in v.items():
#           print(l)
#           if isinstance(m, torch.Tensor):
#               print(l,m.shape)

# sys.exit()

checkpoint_callback_val_loss = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="lowest_val_loss_hsi")
checkpoint_callback_val_miou = ModelCheckpoint(monitor="val_miou", mode="max", save_top_k=1, filename="best_val_miou_hsi")
checkpoint_callback_last_epoch = ModelCheckpoint(monitor="epoch", mode="max", save_top_k=1, filename="last_epoch_hsi")

trainer = L.Trainer(max_epochs=max_num_epochs, accumulate_grad_batches=accumulate_grad_batches, callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=5), checkpoint_callback_val_loss,checkpoint_callback_last_epoch , StochasticWeightAveraging(swa_lrs=swa_lr) ], accelerator="gpu", devices="auto", gradient_clip_val=grad_clip_val,   )  # log_every_n_steps=1, gradient_clip_val=grad_clip_val, min_epochs=min_epochs,( for some reason stops at min epochs) EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=5)
# EarlyStopping(monitor="val_miou", mode="max", verbose=True, patience=5)
    

if training_model == True: 
    
    if tuning_model:
        tuner = Tuner(trainer)


        batch_finder = tuner.scale_batch_size(model, mode="binsearch")

        # print(batch_finder)


        # below can be used to find the lr_for the model
        lr_finder = tuner.lr_find(model)

        # Results can be found in
        # print(lr_finder.results)

        # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        print('suggested lr:', new_lr)

        # update hparams of the model
        model.hparams.learning_rate = new_lr  # learning_rate
        model.hparams.batch_size = batch_finder


        print("learning rate:", model.hparams.learning_rate, "batch size:", model.hparams.batch_size)
        # sys.exit()
        # below trains the model 



        hparams = model.hparams
        with open("hparams_tuned.yaml", "w") as file:
            yaml.dump(hparams, file)
    else:
        model.hparams.learning_rate = initial_lr  # learning_rate
        model.hparams.batch_size = batch_size
        
    #model = torch.compile(model)  # compile code to make it run faster  # this does not work well  due to all the optimizations in place for training 
    trainer.fit(model)


# # Load the model from a checkpoint
# model = DinoV2SemanticSegmentation.load_from_checkpoint("lightning_logs/version_8/checkpoints/last_epoch_hsi.ckpt")
# model = UNET_SemanticSegmentation.load_from_checkpoint("lightning_logs/version_60/checkpoints/lowest_val_loss_hsi.ckpt")
model.eval()
# # test the model on the test set
trainer.test(model)



# # perform inference on a sample image 
# batch = dict()
# batch["hsi_pixel_values"] = test_dataset[0][0].unsqueeze(0)
# batch["rgb_pixel_values"] = test_dataset[0][1].unsqueeze(0)
# batch["labels"] = test_dataset[0][2].unsqueeze(0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# with torch.no_grad():
#     results = model(batch["hsi_pixel_values"].to(device), batch["rgb_pixel_values"].to(device))
# preds = torch.argmax(results, dim=1)


# cmap = 'viridis'

# plt.imshow(preds[0].cpu().numpy(), cmap=cmap)
# plt.colorbar()  
# plt.figure()
# plt.imshow(batch["labels"][0].cpu().numpy(), cmap=cmap)
# plt.colorbar()
# plt.show()


# dinov2 with linear classifier layer results 
#    test_accuracy_mean       0.15635395050048828
#   test_accuracy_overall     0.42304956912994385
#         test_loss           1.9061671495437622
#         test_miou           0.04097169265151024


# dinov2 rgb, unet hyperspectral 
#  test_accuracy_mean       0.21615122258663177
#   test_accuracy_overall     0.5492959022521973
#         test_loss           10.959273338317871
#         test_miou           0.06808633357286453



class DinoV2SemanticSegmentation(L.LightningModule):
        def __init__(self, num_classes,repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=True , tokenW=64, tokenH=64, learning_rate = 1e-3, ignore_index=0):
            super().__init__()
            
            # load the dinov2 model 
            if half_precision:
                self.dinov2 = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
            else:
                self.dinov2= torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
                
            last_layer_params = list(self.dinov2.parameters())[-1]
            patch_descriptor_size = last_layer_params.shape[0]
            
            
            # Freeze the DINOv2 model. This allows for faster training. 
            for _, param in self.dinov2.named_parameters():
                param.requires_grad = False
            
            self.classifier = torch.nn.Conv2d(patch_descriptor_size, num_classes, (1,1))
           
            
            self.patch_descriptor_size = patch_descriptor_size
            self.tokenW = tokenW
            self.tokenH = tokenH
            self.learning_rate = learning_rate
            self.ignore_index = ignore_index
            
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            
            self.train_miou = MeanIoU(num_classes=num_classes, per_class=False)
            self.test_miou = MeanIoU(num_classes=num_classes, per_class=False)
            self.val_miou = MeanIoU(num_classes=num_classes, per_class=False)
            
            self.train_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true", ignore_index=ignore_index)
            self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true", ignore_index=ignore_index)
            self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true", ignore_index=ignore_index)
            
            #  Calculate statistics for each label and average them
            self.train_acc_mean = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=ignore_index)
            self.val_acc_mean = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=ignore_index)
            self.test_acc_mean = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=ignore_index)
            
            #  Sum statistics over all labels
            self.train_acc_overall = MulticlassAccuracy(num_classes=num_classes, average="micro", ignore_index=ignore_index)
            self.val_acc_overall = MulticlassAccuracy(num_classes=num_classes, average="micro", ignore_index=ignore_index)
            self.test_acc_overall = MulticlassAccuracy(num_classes=num_classes, average="micro", ignore_index=ignore_index)
            
            self.save_hyperparameters()
        
        def forward(self, hsi_pixel_values, rgb_pixel_values,):
            # assert not torch.isnan(rgb_pixel_values).any(), "NaN values in input pixel_values"            
            embeddings = self.dinov2.get_intermediate_layers(rgb_pixel_values)[0].squeeze()
            
            assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            embeddings = embeddings.reshape(-1, self.tokenW, self.tokenH, self.patch_descriptor_size)
            embeddings = embeddings.permute(0,3,1,2)
        
            # assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            logits = self.classifier(embeddings)
            # print( logits[0])
            assert not torch.isnan(logits).any(), "NaN values in logits"
            logits = torch.nn.functional.interpolate(logits, size=rgb_pixel_values.shape[2:], mode="bilinear", align_corners=False)
            
            return logits
        
        def log_cf(self, result_cf, step_type):
            
            confusion_matrix_computed = result_cf.detach().cpu().numpy()
            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize = (50,45))
            fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
            plt.close(fig_)
            self.loggers[0].experiment.add_figure(f"Confusion Matrix {step_type}", fig_, self.current_epoch)
            
        def log_data(self, step_type, logits, labels, loss):
            
            preds = torch.argmax(logits, dim=1)
            
            if step_type == "train":
                # result_cf = self.train_confusion_matrix(preds, labels) # not used in training loop
                result_miou = self.train_miou(preds, labels)
                result_acc_overall = self.train_acc_overall(preds, labels)
                results_acc_mean = self.train_acc_mean(preds, labels)
            elif step_type == "val":
                # result_cf = self.val_confusion_matrix(preds, labels)
                result_miou = self.val_miou(preds, labels)
                result_acc_overall = self.val_acc_overall(preds, labels)
                results_acc_mean = self.val_acc_mean(preds, labels)
                # self.log_cf(result_cf, step_type)
            elif step_type == "test":
                # result_cf = self.test_confusion_matrix(preds, labels)
                result_miou = self.test_miou(preds, labels)
                result_acc_overall = self.test_acc_overall(preds, labels)
                results_acc_mean = self.test_acc_mean(preds, labels)
                # self.log_cf(result_cf, step_type)
            else:
                raise ValueError("step_type must be one of 'train', 'val', or 'test'")
            
            self.log(f"{step_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_accuracy_overall", result_acc_overall, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_accuracy_mean", results_acc_mean, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_miou", result_miou, on_step=False, on_epoch=True, prog_bar=True)


        
        def training_step(self, batch, batch_idx):
            
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss = self.loss_fn(logits, labels) 
       
            self.log_data("train", logits, labels, loss)

            return loss
        
        def test_step(self, batch, batch_idx):
            
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits = self.forward(hsi_pixel_values,rgb_pixel_values)
          
            loss = self.loss_fn(logits, labels) 
      
            self.log_data("test", logits, labels, loss)
            
            return loss
        
        def validation_step(self, batch, batch_idx):
            
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits = self.forward(hsi_pixel_values,rgb_pixel_values)
            
            loss = self.loss_fn(logits, labels) 
           
            self.log_data("val", logits, labels, loss)
            
            return loss
        
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
            # return optimizer
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
            return {
            'optimizer': optimizer,
            "lr_scheduler": scheduler
             }
 
 
 
class OneFormer_SemanticSegmentation(L.LightningModule):
        def __init__(self, num_classes, learning_rate = 1e-3, ignore_index=0 ,num_channels=204, processor=None):
            super().__init__()
            
            # shi-labs/oneformer_ade20k_swin_large
            
            # self.rgb_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny",  is_training=True, num_labels = num_classes,  ignore_mismatched_sizes=True)
            self.rgb_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-ade",    num_labels = num_classes,    ignore_mismatched_sizes=True).to(self.device)
            
            # processor.image_processor.num_text = self.rgb_model.config.num_queries - self.rgb_model.config.text_encoder_n_ctx
            self.processor = processor
            
            self.learning_rate = learning_rate
            self.ignore_index = ignore_index
            self.num_classes = num_classes
            
            
            # performance metrics
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            
            self.train_miou = MeanIoU(num_classes=num_classes, per_class=False)
            self.test_miou = MeanIoU(num_classes=num_classes, per_class=False)
            self.val_miou = MeanIoU(num_classes=num_classes, per_class=False)
            
            self.train_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true", ignore_index=ignore_index)
            self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true", ignore_index=ignore_index)
            self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true", ignore_index=ignore_index)
            
            #  Calculate statistics for each label and average them
            self.train_acc_mean = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=ignore_index)
            self.val_acc_mean = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=ignore_index)
            self.test_acc_mean = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=ignore_index)
            
            #  Sum statistics over all labels
            self.train_acc_overall = MulticlassAccuracy(num_classes=num_classes, average="micro", ignore_index=ignore_index)
            self.val_acc_overall = MulticlassAccuracy(num_classes=num_classes, average="micro", ignore_index=ignore_index)
            self.test_acc_overall = MulticlassAccuracy(num_classes=num_classes, average="micro", ignore_index=ignore_index)
            
            
            # UNET 
            #reference
            # https://github.com/hamdaan19/UNet-Multiclass/blob/main/scripts/model.py
            
            # need to modify for dual input
            self.layers = [num_channels, 64, 128, 256, 512, 1024] # use same architecute for 3/200 bands (if higher then it should help to remove some of the noise and take the most information from the bands)
            
            self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
            
            self.up_trans = nn.ModuleList(
                [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
                for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            self.double_conv_ups = nn.ModuleList(
                [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
            self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.final_conv = nn.Conv2d(self.layers[1], num_classes, kernel_size=1)
            
            
            self.fusion_classifier = torch.nn.Sequential(
                nn.ConvTranspose2d(num_classes*2, int(num_classes*2*1.1), kernel_size=7, stride=2), # upsample kernel size 7  since dinov2 has a patch descriptor size of 14x14
                nn.ReLU(),
                nn.Conv2d(int(num_classes*2*1.1), num_classes, kernel_size=7, stride=2)
            )
            
            
            self.save_hyperparameters()
        
        
    
        def forward(self, hsi_pixel_values, rgb_pixel_values, labels):
           

            # # test out unet without rgb right now so just hyperspectral 
            # concat_layers = []
            # x = hsi_pixel_values
            
            # # down layers 
            # for down in self.double_conv_downs:
            #     # x_new = self.adjust_channels(hsi_pixel_values) 
            #     # print(1)
            #     x = down(x)
            #     # print(2)
            #     if down != self.double_conv_downs[-1]:
            #         concat_layers.append(x)
            #         x = self.max_pool_2x2(x)
        
            # concat_layers = concat_layers[::-1]
            
            # # up layers 
            # for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            #     x = up_trans(x)
            #     if x.shape != concat_layer.shape:
            #         x = torchvision.transforms.functional.resize(x, concat_layer.shape[2:])
                
            #     concatenated = torch.cat((concat_layer, x), dim=1)
            #     x = double_conv_up(concatenated)
                
            # x = self.final_conv(x)
            
            
            # replace this with mask2former code 
            

            # if isinstance(labels, list):
            #     labels = torch.tensor(labels)
            # print(self.device)
            batch_tmp = self.processor(rgb_pixel_values, segmentation_maps=labels, return_tensors="pt")
            
            # for key, value in batch_tmp.items():
            #     print(f"{key}: type={type(value)}, shape={value.shape if isinstance(value, torch.Tensor) else 'N/A'}")
                
            # for key, value in batch_tmp.items():
            #     if isinstance(value, list):
            #         batch_tmp[key] = torch.tensor(value)
    
            # # Convert lists to tensors if necessary
            # for key, value in batch_tmp.items():
            #     if isinstance(value, list):
            #         batch_tmp[key] = torch.tensor(value)

            # Move to the appropriate device
            # batch_tmp = {k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in batch_tmp.items()}
            # batch_tmp = batch_tmp.to(self.device)
            
            # Convert lists to tensors if necessary
            # for key, value in batch_tmp.items():
            #     if isinstance(value, list):
            #         batch_tmp[key] = torch.tensor(value)

            # Move to the appropriate device
            # batch_tmp = {key: value.to(self.device) for key, value in batch_tmp.items()}
            semantic_outputs = self.rgb_model(
                    batch_tmp["pixel_values"].float().to(self.device),
                    mask_labels=[labels.to(self.device) for labels in batch_tmp["mask_labels"]],
                    class_labels=[labels.to(self.device) for labels in batch_tmp["class_labels"]]
                )
            
            # semantic_outputs = self.rgb_model(**batch_tmp) #pixel_values = rgb_pixel_values["pixel_values"], pixel_mask = rgb_pixel_values["pixel_mask"], text_inputs = rgb_pixel_values["text_inputs"], task_inputs = rgb_pixel_values["task_inputs"]) # (**rgb_pixel_values)
            
            # pixel_values=batch["pixel_values"].float().to(device),
            #     mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            #     class_labels=[labels.to(device) for labels in batch["class_labels"]],



            
            result = self.processor.post_process_semantic_segmentation(semantic_outputs, target_sizes= [(448,448) for image in batch_tmp["pixel_values"]]) #pixel_mask
            # target_sizes=[rgb_pixel_values.size[::-1]]
            # y = semantic_outputs.masks_queries_logits
            rgb_loss = semantic_outputs.loss
            # print(rgb_loss, "model outputs")
            # print(y.shape)
            result_tensor = torch.stack([torch.tensor(seg_map) for seg_map in result])
            # print(result_tensor.shape)

            
            # # # fuse logits from x and y 
            # combined_embeddings = torch.cat([x, y], dim=1) # concatenate along the channel dimension so descriptors are combined
            # # print(combined_embeddings.shape)
            # z = self.fusion_classifier(combined_embeddings)
            # # print(z.shape)
            return result_tensor, rgb_loss

      
        
        def __double_conv(self, in_channels, out_channels):
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return conv
        
        def log_cf(self, result_cf, step_type):
            
            confusion_matrix_computed = result_cf.detach().cpu().numpy()
            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize = (50,45))
            fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
            plt.close(fig_)
            self.loggers[0].experiment.add_figure(f"Confusion Matrix {step_type}", fig_, self.current_epoch)
            
        def log_data(self, step_type, logits, labels, loss):
            
            preds = logits #torch.argmax(logits, dim=1)
            
            # Check the shapes of preds and labels
            # print(f"Shape of preds: {preds.shape}")
            # print(f"Shape of labels: {labels.shape}")
            # print(" max preds", torch.max(preds), "max labels", torch.max(labels), "min preds", torch.min(preds), "min labels", torch.min(labels))
            assert torch.all(labels >= 0) and torch.all(labels < self.num_classes), "Labels out of bounds"
            assert torch.all(preds >= 0) and torch.all(preds < num_classes), "Predictions out of bounds"

            
    
            
            if step_type == "train":
                # result_cf = self.train_confusion_matrix(preds, labels) # not used in training loop
                result_miou = self.train_miou(preds, labels)
                result_acc_overall = self.train_acc_overall(preds, labels)
                results_acc_mean = self.train_acc_mean(preds, labels)
            elif step_type == "val":
                # result_cf = self.val_confusion_matrix(preds, labels)
                result_miou = self.val_miou(preds, labels)
                result_acc_overall = self.val_acc_overall(preds, labels)
                results_acc_mean = self.val_acc_mean(preds, labels)
                # self.log_cf(result_cf, step_type)
            elif step_type == "test":
                # result_cf = self.test_confusion_matrix(preds, labels)
                result_miou = self.test_miou(preds, labels)
                result_acc_overall = self.test_acc_overall(preds, labels)
                results_acc_mean = self.test_acc_mean(preds, labels)
                # self.log_cf(result_cf, step_type)
            else:
                raise ValueError("step_type must be one of 'train', 'val', or 'test'")
            
            self.log(f"{step_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_accuracy_overall", result_acc_overall, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_accuracy_mean", results_acc_mean, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_miou", result_miou, on_step=False, on_epoch=True, prog_bar=True)


        
        def training_step(self, batch, batch_idx):
            
            step_type = "train"
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            result, rgb_loss = self.forward(hsi_pixel_values,rgb_pixel_values, labels)
            # loss_hsi = self.loss_fn(logits_hsi, labels) 
            # loss_fused = self.loss_fn(logits_fused, labels)
       
            # combined_loss = loss_hsi + loss_rgb + 4*loss_fused # penalize fused loss a lot more since it is final output
            
            if rgb_loss == None:
                rgb_loss = torch.tensor(10.0, device=self.device)  # Ensure rgb_loss is a Tensor
                print("Warning: rgb_loss is None")
            self.log_data(step_type, result, labels, rgb_loss)

            
            return rgb_loss
        
        def test_step(self, batch, batch_idx):
            
            step_type = "test"
            
            # rgb_pixel_values = batch["rgb_pixel_values"]
            # hsi_pixel_values = batch["hsi_pixel_values"]
            # labels = batch["labels"]     
            
            # logits_hsi, logits_rgb, logits_fused  = self.forward(hsi_pixel_values,rgb_pixel_values)
            # loss_hsi = self.loss_fn(logits_hsi, labels) 
            # loss_rgb = self.loss_fn(logits_rgb, labels)
            # loss_fused = self.loss_fn(logits_fused, labels)
       
            # combined_loss = loss_hsi + loss_rgb + 4*loss_fused
            
            
            # self.log_data(step_type, logits_fused, labels, combined_loss)

            
            # return combined_loss
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            result, rgb_loss = self.forward(hsi_pixel_values,rgb_pixel_values)
            # loss_hsi = self.loss_fn(logits_hsi, labels) 
            # loss_fused = self.loss_fn(logits_fused, labels)
       
            # combined_loss = loss_hsi + loss_rgb + 4*loss_fused # penalize fused loss a lot more since it is final output
            
            if rgb_loss == None:
                rgb_loss = 10
                print("Warning: rgb_loss is None")
            
            self.log_data(step_type, result, labels, rgb_loss)

            
            return rgb_loss
        
        def validation_step(self, batch, batch_idx):
            
            step_type = "val"
            
            # rgb_pixel_values = batch["rgb_pixel_values"]
            # hsi_pixel_values = batch["hsi_pixel_values"]
            # labels = batch["labels"]     
            
            # logits_hsi, logits_rgb, logits_fused = self.forward(hsi_pixel_values,rgb_pixel_values)
            # loss_hsi = self.loss_fn(logits_hsi, labels) 
            # loss_rgb = self.loss_fn(logits_rgb, labels)
            # loss_fused = self.loss_fn(logits_fused, labels)
       
            # combined_loss = loss_hsi + loss_rgb + 4*loss_fused
            
            
            # self.log_data(step_type, logits_fused, labels, combined_loss)

            
            # return combined_loss
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            result, rgb_loss = self.forward(hsi_pixel_values,rgb_pixel_values, labels)
            # loss_hsi = self.loss_fn(logits_hsi, labels) 
            # loss_fused = self.loss_fn(logits_fused, labels)
       
            # combined_loss = loss_hsi + loss_rgb + 4*loss_fused # penalize fused loss a lot more since it is final output
            if rgb_loss == None:
                rgb_loss = 10
                print("Warning: rgb_loss is None")
            
            self.log_data(step_type, result, labels, rgb_loss)

            
            return rgb_loss
        
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
            # return optimizer
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
            return {
            'optimizer': optimizer,
            "lr_scheduler": scheduler
             }       


