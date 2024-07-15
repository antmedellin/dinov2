# import dependencies 

import os
from os.path import join
from torch.utils.data import Dataset
import torch
import albumentations as A 
from PIL import Image
import numpy as np
import json
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2
from osgeo import gdal
from torch.utils.data import DataLoader
from transformers.modeling_outputs import SemanticSegmenterOutput
import evaluate
from torch.optim import AdamW
from tqdm.auto import tqdm
from PIL import Image
import builtins
import functools
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import torch.nn as nn
import torch.nn.functional as F


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
  
  
class SegmentationDataset(Dataset):
    def __init__(self, image_set,  root_dir, id2color, transform=None):
        
        # image_set # train ,test, validation
        self.transform = transform
        self.root = join(root_dir,image_set)
        
        
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
        

        
        return hsi_img, rgb_img, label_img_greyscale


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

def collate_fn(inputs):


    batch = dict()
    batch["hsi_pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["rgb_pixel_values"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[2] for i in inputs], dim=0)

    return batch

class LinearClassifierNew(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifierNew, self).__init__()

        self.in_channels = in_channels # patch descriptor size
        self.width = tokenW
        self.height = tokenH
        self.num_labels = num_labels
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))
        # self.classifier = torch.nn.Linear(in_channels, num_labels)


    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)
        results = self.classifier(embeddings)

        return results
    
    
class DinoV2SemanticSegmentationRegisters(torch.nn.Module):
    def __init__(self,  num_labels=1, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg",   half_precision=False, device="cuda"):
        super().__init__()
        self.repo_name = repo_name
        self.model_name = model_name
        self.half_precision = half_precision
        self.device = device
        
        # load the dinov2 model 
        if self.half_precision:
            self.dinov2 = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.dinov2= torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
        
        # Get the parameters of the last layer
        last_layer_params = list(self.dinov2.parameters())[-1]
        # get the patch descriptor size for use in initializing the linear classifier layer 
        patch_descriptor_size = last_layer_params.shape[0]
        
        # Create the classifier that will be used for semantic segmentation on top of the DINOv2 model
        self.classifier = LinearClassifierNew(patch_descriptor_size, 32, 32, num_labels)
        self.classifier = self.classifier.to(self.device)
        
        # Freeze the DINOv2 model. This allows for faster training. 
        for _, param in self.dinov2.named_parameters():
            param.requires_grad = False
            
            
            
    def __call__(self, pixel_values, labels=None):
        pixel_values = pixel_values.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        return self.forward(pixel_values=pixel_values, labels=labels)         
        
    def forward(self, pixel_values, labels=None):
        # Get the embeddings from the DINOv2 model which are patch descriptors
        patch_embeddings = self.dinov2.get_intermediate_layers(pixel_values)[0].squeeze()
        
        # print("patch embeddings shape: ", patch_embeddings.shape)
            
        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        # print("logits shape: ", logits.shape)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index 
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fct(logits, labels)
        
        return  SemanticSegmenterOutput(
            loss=loss,
            logits=logits
        )
        
# hyperspectral semantic segmentation model
class HSIClassifier(torch.nn.Module):
    def __init__(self, device="cuda", num_labels=1,):
        super(HSIClassifier, self, ).__init__()
        
        self.device = device

        self.classifier = LinearClassifierNew(768, 32, 32, num_labels)
        self.classifier = self.classifier.to(self.device)

        self.descriptor =  torch.nn.Sequential(
            
            torch.nn.Conv2d(num_channels, 256, kernel_size=3, padding=1),  # Adjusted for input size
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),  # 224x224
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),  # 112x112
            torch.nn.Conv2d(512, 768, kernel_size=3, padding=2),  # Adjusted for output channels
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(2, 2),  # 56x56
            torch.nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=2),  # Additional layer
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(2, 2),  # 28x28
            torch.nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=3),  # Additional layer to reach 32x32
            # torch.nn.LeakyReLU(),
            # No pooling here to maintain 32x32 size
            torch.nn.Flatten(start_dim=2),
            
        )
        # assume input is 448x448x204
        # we want result to be 32x32x768 to match dinov2 vitb descriptor size 
        
        # 448 / 32 = 14 so each patch is 14x14 pixels
        # 32 * 32 = 1024 patches this way we can use the same code for both approaches 
        
        
    def __call__(self, pixel_values, labels=None):
        pixel_values = pixel_values.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        return self.forward(pixel_values=pixel_values, labels=labels)  

    def forward(self,  pixel_values, labels=None):
        
        patch_embeddings = self.descriptor(pixel_values)
        # print (patch_embeddings.shape, "patch_embeddings shape hsi")
        logits = self.classifier(patch_embeddings)
        # logits = logits.permute(0,2,1)
        # print (logits.shape, "logits shape hsi")
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index 
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fct(logits, labels)
        
        return  SemanticSegmenterOutput(
            loss=loss,
            logits=logits
        )
# combined hsi and rgb model
class HSItestClassifier(torch.nn.Module):
    def __init__(self,  num_labels=1, device="cuda"):
        super(HSItestClassifier, self).__init__()
        
        self.device = device     
            
        self.hsi_classifier = LinearClassifierNew(768, 32, 32, num_labels)
        self.hsi_classifier = self.hsi_classifier.to(self.device)
        
        self.hsi_model =  torch.nn.Sequential(
            
            torch.nn.Conv2d(num_channels, 256, kernel_size=3, padding=1),  
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),  #
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),  
            torch.nn.Conv2d(512, 768, kernel_size=3, padding=2),  
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=2),  
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=3),  
            torch.nn.Flatten(start_dim=2),
            
        )
        
        self.hsi_model_mlp =  torch.nn.Sequential(
            #204 input channels, 448, 448 input size
            torch.nn.Linear(204,512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(512,512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(512,512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(512,num_labels),
            

            
        )
        

        
    def forward(self, hsi_pixel_values, rgb_pixel_values, labels=None):
        
        
        
        # hsi_embeddings = self.hsi_model(hsi_pixel_values)
        # if hsi_embeddings.shape[0] == 1:
        #     hsi_embeddings = hsi_embeddings.squeeze(0)
        # hsi_embeddings = hsi_embeddings.transpose(0,1)
        # hsi_logits = self.hsi_classifier(hsi_embeddings)
        
        hsi_pixel_values = hsi_pixel_values.permute(0,2,3,1)
        
        hsi_logits = self.hsi_model_mlp(hsi_pixel_values)
        # print(hsi_logits.shape, "hsi_logits shape")
        
        hsi_logits = hsi_logits.permute(0, 3, 1, 2)  # Now hsi_logits_permuted has shape [1, 45, 448, 448]

        # hsi_logits2 = torch.nn.functional.interpolate(hsi_logits, size=hsi_pixel_values.shape[2:], mode="bilinear", align_corners=False)
     
        # print(hsi_logits.shape, "hsi_logits shape", labels.shape, "labels shape")

     
     
     
        hsi_loss = None
        
        if labels is not None:
            # important: we're going to use 0 here as ignore index 
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            # loss_fct = FocalLoss(ignore_index=ignore_index)

            hsi_loss = loss_fct(hsi_logits, labels)
           

            
            return  SemanticSegmenterOutput(
                
                loss= hsi_loss  ,
                logits=hsi_logits
            )
        else:
            return  SemanticSegmenterOutput(
                loss= hsi_loss ,
                logits=hsi_logits
            )
        
    
        
        
    def __call__(self, hsi_pixel_values, rgb_pixel_values,  labels=None):
        hsi_pixel_values = hsi_pixel_values.to(self.device)

        if labels is not None:
            labels = labels.to(self.device)
        return self.forward(hsi_pixel_values=hsi_pixel_values, rgb_pixel_values=rgb_pixel_values, labels=labels)        
 
# combined hsi and rgb model
class RGBclassifier(torch.nn.Module):
    def __init__(self,  num_labels=1, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg",   half_precision=False, device="cuda"):
        super(RGBclassifier, self).__init__()
        
        self.device = device     
        
        self.repo_name = repo_name
        self.model_name = model_name
        self.half_precision = half_precision
        self.device = device
        
        # load the dinov2 model 
        if self.half_precision:
            self.rgb_model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.rgb_model= torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
            
        # Get the parameters of the last layer
        last_layer_params = list(self.rgb_model.parameters())[-1]
        # get the patch descriptor size for use in initializing the linear classifier layer 
        patch_descriptor_size = last_layer_params.shape[0]    
        
        # Freeze the DINOv2 model. This allows for faster training. 
        for _, param in self.rgb_model.named_parameters():
            param.requires_grad = False
            
        
        self.rgb_classifier = torch.nn.Sequential(
            #204 input channels, 448, 448 input size
            torch.nn.Linear(768,768),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(768,768),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(768,512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(512,num_labels),
        )
        
        
        self.rgb_classifier = self.rgb_classifier.to(self.device)
        
        
        self.classifier = torch.nn.Sequential(
                    torch.nn.Conv2d(768, 512, (1,1)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(512, 256, (1,1)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(256, 128, (1,1)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(128, 64, (1,1)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(64, num_labels, (1,1))



        )
        
        
        self.classifier = self.classifier.to(self.device)
        
        
        
        
        self.num_labels = num_labels

        
    def forward(self, hsi_pixel_values, rgb_pixel_values, labels=None):
        
        
        rgb_embeddings = self.rgb_model.get_intermediate_layers(rgb_pixel_values)[0].squeeze()
        
        
        # rgb_logits = self.rgb_classifier(rgb_embeddings)
        # rgb_logits.transpose(0,1)
        # rgb_logits = rgb_logits.reshape(1, self.num_labels, 32,32)
        
        rgb_embeddings = rgb_embeddings.reshape(-1, 32,32, 768)
        rgb_embeddings = rgb_embeddings.permute(0,3,1,2)
        
        rgb_embeddings = torch.nn.functional.interpolate(rgb_embeddings, size=hsi_pixel_values.shape[2:], mode="bilinear", align_corners=False)
        
        rgb_logits = self.classifier(rgb_embeddings)
        
        
        # print(rgb_logits.shape, "rgb_logits shape", rgb_embeddings.shape, "rgb_embeddings shape")
        # loss of rgb and hsi logits
        rgb_logits = torch.nn.functional.interpolate(rgb_logits, size=hsi_pixel_values.shape[2:], mode="bilinear", align_corners=False)
        
        
        rgb_loss = None
  
        
        if labels is not None:
            # important: we're going to use 0 here as ignore index 
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            # loss_fct = FocalLoss(ignore_index=ignore_index)

            rgb_loss = loss_fct(rgb_logits, labels)
 
            
            return  SemanticSegmenterOutput(
                
                loss=rgb_loss , # combined loss is weighted more since it is the combined model and has the output logits
                logits=rgb_logits
            )
        else:
            return  SemanticSegmenterOutput(
                loss=rgb_loss ,
                logits=rgb_logits
            )
            
            
                     
# combined hsi and rgb model
class CombinedClassifier(torch.nn.Module):
    def __init__(self,  num_labels=1, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg",   half_precision=False, device="cuda"):
        super(CombinedClassifier, self).__init__()
        
        self.device = device     
        
        self.repo_name = repo_name
        self.model_name = model_name
        self.half_precision = half_precision
        self.device = device
        
        # load the dinov2 model 
        if self.half_precision:
            self.rgb_model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.rgb_model= torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
            
        # Get the parameters of the last layer
        last_layer_params = list(self.rgb_model.parameters())[-1]
        # get the patch descriptor size for use in initializing the linear classifier layer 
        patch_descriptor_size = last_layer_params.shape[0]    
        
        # Freeze the DINOv2 model. This allows for faster training. 
        for _, param in self.rgb_model.named_parameters():
            param.requires_grad = False
            
        self.rgb_classifier = LinearClassifierNew(patch_descriptor_size, 32, 32, num_labels)
        self.hsi_classifier = LinearClassifierNew(768, 32, 32, num_labels)
        
        self.rgb_classifier = self.rgb_classifier.to(self.device)
        self.hsi_classifier = self.hsi_classifier.to(self.device)
        
        self.hsi_model =  torch.nn.Sequential(
            
            torch.nn.Conv2d(num_channels, 256, kernel_size=3, padding=1),  
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),  #
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),  
            torch.nn.Conv2d(512, 768, kernel_size=3, padding=2),  
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=2),  
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=3),  
            torch.nn.Flatten(start_dim=2),
            
        )
        
        self.fusion_classifier = LinearClassifierNew(num_labels*2, 32, 32, num_labels)
        self.fusion_classifier = self.fusion_classifier.to(self.device)

        
        
    def forward(self, hsi_pixel_values, rgb_pixel_values, labels=None):
        
        
        rgb_embeddings = self.rgb_model.get_intermediate_layers(rgb_pixel_values)[0].squeeze()
        
        rgb_logits = self.rgb_classifier(rgb_embeddings)
        
        hsi_embeddings = self.hsi_model(hsi_pixel_values)
        
        if hsi_embeddings.shape[0] == 1:
            hsi_embeddings = hsi_embeddings.squeeze(0)
        hsi_embeddings = hsi_embeddings.transpose(0,1)
        
        
        hsi_logits = self.hsi_classifier(hsi_embeddings)
        
        # # print(rgb_logits.shape, hsi_logits.shape, rgb_embeddings.shape, hsi_embeddings.shape)
        
        combined_embeddings = torch.cat([rgb_logits, hsi_logits], dim=1) # concatenate along the channel dimension so descriptors are combined

        combined_logits = self.fusion_classifier(combined_embeddings)
        
        # print(combined_logits.shape,  combined_embeddings.shape)

        
        # test out using embeddings next instead of the logits
        
        # loss of rgb and hsi logits
        rgb_logits = torch.nn.functional.interpolate(rgb_logits, size=hsi_pixel_values.shape[2:], mode="bilinear", align_corners=False)
        hsi_logits = torch.nn.functional.interpolate(hsi_logits, size=hsi_pixel_values.shape[2:], mode="bilinear", align_corners=False)
        combined_logits = torch.nn.functional.interpolate(combined_logits, size=hsi_pixel_values.shape[2:], mode="bilinear", align_corners=False)
        
        rgb_loss = None
        hsi_loss = None
        combined_loss = None
        
        if labels is not None:
            # important: we're going to use 0 here as ignore index 
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            # loss_fct = FocalLoss(ignore_index=ignore_index)

            rgb_loss = loss_fct(rgb_logits, labels)
            hsi_loss = loss_fct(hsi_logits, labels)
            combined_loss = loss_fct(combined_logits, labels)
           

            
            return  SemanticSegmenterOutput(
                
                loss=rgb_loss + hsi_loss + 2*combined_loss, # combined loss is weighted more since it is the combined model and has the output logits
                logits=combined_logits
            )
        else:
            return  SemanticSegmenterOutput(
                loss=rgb_loss + hsi_loss + 2*combined_loss,
                logits=combined_logits
            )
        
    
        
        
    def __call__(self, hsi_pixel_values, rgb_pixel_values,  labels=None):
        rgb_pixel_values = rgb_pixel_values.to(self.device)
        hsi_pixel_values = hsi_pixel_values.to(self.device)

        if labels is not None:
            labels = labels.to(self.device)
        return self.forward(hsi_pixel_values=hsi_pixel_values, rgb_pixel_values=rgb_pixel_values, labels=labels)        


# Function to update learning rate
def update_learning_rate(optimizer, epoch, warmup_lr, base_lr, num_warmup_epochs):
    if epoch < num_warmup_epochs:
        # Linear warm-up
        lr = warmup_lr + (base_lr - warmup_lr) * (epoch / num_warmup_epochs)
    else:
        lr = base_lr  # Keep constant after warm-up, or implement your schedule
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        
#clear cuda memory
torch.cuda.empty_cache()

# set variables 
REPO_NAME = "facebookresearch/dinov2"
MODEL_NAME = "dinov2_vitb14_reg"
dataset_dir = "/workspaces/LIB-HSI"
rgb_data_json = '/workspaces/dinov2/notebooks/lib_hsi_rgb.json'
output_dir = "/workspaces/dinov2/output"


epochs = 50
num_warmup_epochs = 5
batch_size = 1
num_workers = 6
ignore_index=-1

initial_lr = 0.0001  # Initial learning rate for warm-up
base_lr = 0.001  # Learning rate after warm-up
warmup_lr = initial_lr

early_stopping_patience = 10
early_stopping_min_epoch = 20


train_transform = A.Compose([
    
    A.RandomCrop(width=400, height=400),  # Example of spatial augmentation
    A.HorizontalFlip(p=0.5),  # Example of flip augmentation
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Example of color augmentation
    A.Rotate(limit=45, p=0.5), 
    A.Resize(width=448, height=448),
])


base_transform = A.Compose([
    A.Resize(width=448, height=448),
])

# Redefine the print function to automatically flush by default
builtins.print = functools.partial(print, flush=True)

file_data =  open(rgb_data_json)
file_contents = json.load(file_data)

id2label ={}
id2color = {}
for i, item in enumerate(file_contents['items'], start=0):
    id2label[i] = item['name']
    id2color[i] = [item['red_value'], item['green_value'], item['blue_value']]
    
print(id2label)
print(id2color)
num_classes = len(id2label)
print("num classes",num_classes)

train_dataset = SegmentationDataset(image_set="train", root_dir=dataset_dir, id2color=id2color, transform=train_transform)

test_dataset = SegmentationDataset(image_set="test", root_dir=dataset_dir, id2color=id2color,  transform=base_transform)

# test_dataset = SegmentationDataset(image_set="validation", root_dir=dataset_dir, id2color=id2color, transform=base_transform)
# train_dataset = SegmentationDataset(image_set="validation", root_dir=dataset_dir, id2color=id2color, transform=base_transform)

val_dataset = SegmentationDataset(image_set="validation", root_dir=dataset_dir, id2color=id2color, transform=base_transform)

print("train dataset size: ", train_dataset.__len__())
print("test dataset size: ", test_dataset.__len__())
print("validation dataset size: ", val_dataset.__len__())

hsi, rgb, mask = train_dataset[0]
num_channels = hsi.shape[0]

# dataloader allows us to get batches of data from the datasets
# batch size and number of workers can be modified to better suit the specs of your machine
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=num_workers)
val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=num_workers)


# model = DinoV2SemanticSegmentationRegisters(num_labels = num_classes, repo_name=REPO_NAME, model_name=MODEL_NAME, half_precision=False, device="cuda")  

# model = CombinedClassifier(num_labels = num_classes, repo_name=REPO_NAME, model_name=MODEL_NAME, half_precision=False, device="cuda")

# model = HSItestClassifier(num_labels = num_classes, device="cuda") 

model = RGBclassifier(num_labels = num_classes, repo_name=REPO_NAME, model_name=MODEL_NAME, half_precision=False, device="cuda")



#initialize metrics for model
metric = evaluate.load("mean_iou")
metric_val = evaluate.load("mean_iou")

# set optimizer
optimizer = AdamW(model.parameters(), lr=initial_lr)

# set device for processing and move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("using" , device)


# initialize empty data stuctures to keep track of learning performance

history_loss_train = []
history_loss_val = []

history_mean_iou_train = []
history_mean_iou_val = []

history_mean_accuracy_train = []
history_mean_accuracy_val = []

history_overall_accuracy_train = []
history_overall_accuracy_val = []

highest_accuracy = 0
lowest_loss = 1000000
highest_iou = 0
highest_overall_accuracy = 0
epochs_since_improvement = 0
early_stop = False
# best_score = float('inf')
best_score = 0



# Create a directory to save the best models
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# save model in sub directory named model
model_directory = os.path.join(output_dir, "model")
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
   

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)



# start training
for epoch in range(epochs):
    # print("Epoch:", epoch+1)
    current_lr = update_learning_rate(optimizer, epoch, warmup_lr, base_lr, num_warmup_epochs)
    
    if epoch > num_warmup_epochs:
        scheduler.step(history_loss_val[-1])
    print(f"Epoch: {epoch+1}, Learning Rate: {current_lr}")
    
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader, file=sys.stdout)):
        torch.cuda.synchronize()
        # pixel_values = batch["rgb_pixel_values"].to(device)
        # pixel_values = batch["hsi_pixel_values"].to(device)
        rgb_pixel_values = batch["rgb_pixel_values"].to(device)
        hsi_pixel_values = batch["hsi_pixel_values"].to(device)

        labels = batch["labels"].to(device).long()

        # forward pass
        # outputs = model(pixel_values, labels=labels)
        outputs = model(hsi_pixel_values, rgb_pixel_values,  labels=labels)
        loss = outputs.loss

        # zero the parameter gradients
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            predicted = outputs.logits.argmax(dim=1)

            # note that the metric expects predictions + labels as numpy arrays
            
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
    
    

  
    metrics = metric.compute(num_labels=num_classes,
                                ignore_index=ignore_index,
                                reduce_labels=False,
    )
    history_loss_train.append(loss.item())
    history_mean_iou_train.append(metrics["mean_iou"])
    history_mean_accuracy_train.append(metrics["mean_accuracy"])
    history_overall_accuracy_train.append(metrics["overall_accuracy"])
    
    model.eval()
    val_loss_accumulated = 0  # Initialize variable to accumulate validation loss

    for idx, batch in enumerate(tqdm(val_dataloader, file=sys.stdout)):
        # pixel_values = batch["rgb_pixel_values"].to(device)
        # pixel_values = batch["hsi_pixel_values"].to(device)
        hsi_pixel_values = batch["hsi_pixel_values"].to(device)
        rgb_pixel_values = batch["rgb_pixel_values"].to(device)

        labels = batch["labels"].to(device).long()


        # forward pass
        # outputs = model(pixel_values, labels=labels)
        outputs = model(hsi_pixel_values, rgb_pixel_values, labels=labels)
        val_loss = outputs.loss
        val_loss_accumulated += val_loss.item()  

        with torch.no_grad():
            predicted = outputs.logits.argmax(dim=1)

            # note that the metric expects predictions + labels as numpy arrays
            
            metric_val.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
   
    average_val_loss = val_loss_accumulated / len(val_dataloader)

    metrics_val = metric_val.compute(num_labels=num_classes,
                            ignore_index=ignore_index,
                            reduce_labels=False,
    )
    history_loss_val.append(average_val_loss)
    history_mean_iou_val.append(metrics_val["mean_iou"])
    history_mean_accuracy_val.append(metrics_val["mean_accuracy"])
    history_overall_accuracy_train.append(metrics_val["overall_accuracy"])
    
    print("Train Loss: ", loss.item(), " Validation Loss: ", average_val_loss)
    print("Train Mean_iou: ", metrics["mean_iou"], " Validation Mean_iou: ", metrics_val["mean_iou"])
    print("Train Mean_accuracy: ", metrics["mean_accuracy"], " Validation Mean_accuracy: ", metrics_val["mean_accuracy"])
    print("Train Overall_accuracy: ", metrics["overall_accuracy"], " Validation Overall_accuracy: ", metrics_val["overall_accuracy"])
    
    
    if metrics_val["mean_accuracy"] > highest_accuracy:
        highest_accuracy = metrics_val["mean_accuracy"]
        torch.save(model.state_dict(), os.path.join(model_directory, "highest_accuracy_model.pth"))
        
    if average_val_loss < lowest_loss:
        lowest_loss = average_val_loss
        torch.save(model.state_dict(), os.path.join(model_directory, "lowest_loss_model.pth"))
        
    if metrics_val["mean_iou"] > highest_iou:
        highest_iou = metrics_val["mean_iou"]
        torch.save(model.state_dict(), os.path.join(model_directory, "highest_iou_model.pth"))
        
    if metrics_val["overall_accuracy"] > highest_overall_accuracy:
        highest_overall_accuracy = metrics_val["overall_accuracy"]
        torch.save(model.state_dict(), os.path.join(model_directory, "highest_overall_accuracy_model.pth"))
        
    if average_val_loss < best_score: # use loss as the metric to compare rather than accuracy
    # if highest_accuracy > best_score: # use accuracy as the metric to compare rather than loss
        best_score = average_val_loss
        # best_score = highest_accuracy
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1
        if epochs_since_improvement == early_stopping_patience and epoch > early_stopping_min_epoch:
            print("Early stopping")
            break
if not early_stop:
    print("Completed all epochs without early stopping.")


result_directory = os.path.join(output_dir, "results")
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
    
# load the best model and save the results
model.load_state_dict(torch.load(os.path.join(model_directory, "highest_accuracy_model.pth")))
model.to(device)
model.eval()

test_metrics = evaluate.load("mean_iou")

for idx in range(test_dataset.__len__()):
# for idx in range(test_dataset.__len__()):
    # in each directory save ground truth image, rgb image, and predicted image
    
    hsi_image, image, mask = test_dataset[idx]
    img_permute = image.permute(1, 2, 0)
    rgb_pixel_values = torch.tensor(image)
    hsi_pixel_values = torch.tensor(hsi_image)
    rgb_pixel_values = rgb_pixel_values.unsqueeze(0) # convert to (batch_size, num_channels, height, width)
    hsi_pixel_values = hsi_pixel_values.unsqueeze(0) # convert to (batch_size, num_channels, height, width)
    with torch.no_grad():
        outputs = model(hsi_pixel_values.to(device), rgb_pixel_values.to(device))
        
    test_metrics.add_batch(predictions=outputs.logits.argmax(dim=1).detach().cpu().numpy(), references=mask.unsqueeze(0).detach().cpu().numpy())
    test_metrics.compute(num_labels=num_classes, ignore_index=ignore_index, reduce_labels=False)
        
    upsampled_logits = torch.nn.functional.interpolate(outputs.logits, size=(image.shape[-2], image.shape[-1]), mode="bilinear",align_corners=False)
    
    predicted_map = upsampled_logits.argmax(dim=1)
    
    
    result_idx_directory = os.path.join(result_directory, str(idx))
    if not os.path.exists(result_idx_directory):
        os.makedirs(result_idx_directory)
        
    mask_color = mask.detach().cpu().numpy()
    color_mask = np.zeros((mask_color.shape[0], mask_color.shape[1], 3), dtype=np.uint8)
    # convert id to rgb values in mask
    for i in range(mask_color.shape[0]):
            for j in range(mask_color.shape[1]):                      
                pixel_value = int(mask_color[i, j])
                color_mask[i,j,:] = id2color[pixel_value]
                
    predicted_mask = predicted_map.squeeze().detach().cpu().numpy()
    color_predicted_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    # convert id to rgb values in mask
    for i in range(predicted_mask.shape[0]):
            for j in range(predicted_mask.shape[1]):                      
                pixel_value = int(predicted_mask[i, j])
                color_predicted_mask[i,j,:] = id2color[pixel_value]
    
    
    # save the ground truth image
    cv2.imwrite(os.path.join(result_idx_directory, "ground_truth.png"), color_mask)
    # save the rgb image
    cv2.imwrite(os.path.join(result_idx_directory, "rgb_image.png"), img_permute.detach().cpu().numpy())
    # save the predicted image
    cv2.imwrite(os.path.join(result_idx_directory, "predicted_image.png"), color_predicted_mask)
    print("Saved results for image: ", idx)
    
        
    

print("Test Mean_iou: ", test_metrics["mean_iou"], " Test Mean_accuracy: ", test_metrics["mean_accuracy"], " Test Overall_accuracy: ", test_metrics["overall_accuracy"])

print("finished")