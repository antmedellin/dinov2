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
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics.segmentation import MeanIoU
import torch.optim.lr_scheduler as lr_scheduler 
from lightning.pytorch.callbacks import ModelCheckpoint
import cv2
import pandas as pd
from osgeo import gdal
import json
import seaborn as sns
import torchvision 
import sys
from PIL import Image
import matplotlib
# matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt  
import functools
import builtins
builtins.print = functools.partial(print, flush=True) 
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
# pip install segmentation-models-pytorch

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
    
class LIBHSIDataset(Dataset):
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
            transformed = self.transform(image = rgb_img, mask = label_img_greyscale, hsi_image = hsi_img)
        
            hsi_img, rgb_img, label_img_greyscale = torch.tensor(transformed['hsi_image']), torch.tensor(transformed['image']), torch.tensor(transformed['mask'])
        else:
            hsi_img, rgb_img, label_img_greyscale = torch.tensor(hsi_img), torch.tensor(rgb_img), torch.tensor(label_img_greyscale)
            
            
        #convert from x,y,channels to channels, x, y
        hsi_img = hsi_img.permute(2,0,1)
        rgb_img = rgb_img.permute(2,0,1)
        
        #convert from uint8 to float32
        hsi_img = hsi_img.float()
        rgb_img = rgb_img.float()
            
        return hsi_img, rgb_img, label_img_greyscale   
    
def collate_fn(inputs):

    batch = dict()
    batch["hsi_pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["rgb_pixel_values"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[2] for i in inputs], dim=0).long()

    return batch   
    
class UNET_SemanticSegmentation(L.LightningModule):
        def __init__(self, num_classes, learning_rate = 1e-3, ignore_index=0 ,num_channels=204, num_workers=4, train_dataset=None, val_dataset=None, test_dataset = None, batch_size=2 ):
            super().__init__()
            
            self.learning_rate = learning_rate
            # self.batch_size = batch_size override in dataloaders
            self.ignore_index = ignore_index
            self.num_workers = num_workers
            self.num_classes = num_classes
            self.num_channels = num_channels
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
            
            self.save_hyperparameters()
            
            self.loss_fn = CombinedLoss(ignore_index=self.ignore_index)
            
            self.hsi_unet = smp.Unet( 'resnet152', in_channels=self.num_channels, classes=self.num_classes, encoder_depth=5)

            self.train_miou = MeanIoU(num_classes=self.num_classes, per_class=False)
            self.test_miou = MeanIoU(num_classes=self.num_classes, per_class=False)
            self.val_miou = MeanIoU(num_classes=self.num_classes, per_class=False)
            
            self.train_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize="true", ignore_index=self.ignore_index)
            self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize="true", ignore_index=self.ignore_index)
            self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize="true", ignore_index=self.ignore_index)
            
            #  Calculate statistics for each label and average them
            self.train_acc_mean = MulticlassAccuracy(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index)
            self.val_acc_mean = MulticlassAccuracy(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index)
            self.test_acc_mean = MulticlassAccuracy(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index)
            
            #  Sum statistics over all labels
            self.train_acc_overall = MulticlassAccuracy(num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index)
            self.val_acc_overall = MulticlassAccuracy(num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index)
            self.test_acc_overall = MulticlassAccuracy(num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index)
        
        def forward(self, hsi_pixel_values, rgb_pixel_values):
            
            x = self.hsi_unet(hsi_pixel_values)
            
            return x
        
        def log_cf(self, result_cf, step_type):
            
            confusion_matrix_computed = result_cf.detach().cpu().numpy()
            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize = (self.num_classes+5,self.num_classes))
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
            
            self.log(f"{step_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_accuracy_overall", result_acc_overall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_accuracy_mean", results_acc_mean, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_miou", result_miou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        def training_step(self, batch, batch_idx):
            
            step_type = "train"
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits_hsi = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss_hsi = self.loss_fn(logits_hsi, labels) 
            
            self.log_data(step_type, logits_hsi, labels, loss_hsi)

            return loss_hsi
        
        def test_step(self, batch, batch_idx):
            
            step_type = "test"
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits_hsi = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss_hsi = self.loss_fn(logits_hsi, labels) 
            
            self.log_data(step_type, logits_hsi, labels, loss_hsi)

            return loss_hsi
        
        def validation_step(self, batch, batch_idx):
                
            step_type = "val"
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits_hsi = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss_hsi = self.loss_fn(logits_hsi, labels) 
            
            self.log_data(step_type, logits_hsi, labels, loss_hsi)

            return loss_hsi
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
            # return optimizer
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
            return {
            'optimizer': optimizer,
            "lr_scheduler": scheduler
             }     
              
        def train_dataloader(self):
            
            return  DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn,num_workers=self.num_workers)
        
        def val_dataloader(self):
            
            return  DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=self.num_workers)
        
        def test_dataloader(self):
            
            return  DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=self.num_workers)


dataset_dir='/workspaces/LIB-HSI'
rgb_data_json = '/workspaces/dinov2/notebooks/lib_hsi_rgb.json'

batch_size = 2
# ignore_index=0 #-1
ignore_index=7 # misc. class, 
num_workers = 3 #  os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() is None
initial_lr =  0.0001 #0.00001 
swa_lr = 0.01
# these should be multiple of 14 for dino model 
img_height = 448
img_width = 448
max_num_epochs = 100
accumulate_grad_batches = 5# 5 # increases the effective batch size  # 1 means no accumulation # more important when batch size is small or not doing multi gpu training
grad_clip_val = 5 # clip gradients that have norm bigger than this
training_model = True
tuning_model = True
min_epochs = 20

# Define mean and standard deviation for normalization
# Use the same value for all channels
num_channels = 204
mean = [0.45] * num_channels  
std = [0.225] * num_channels   

torch.cuda.empty_cache()

test_transform = A.Compose([
    # A.Resize(width=img_width, height=img_height), 
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0)

], additional_targets={"hsi_image": "image"})

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.RandomCrop(width=450, height=450),
    # A.Resize(width=img_width, height=img_height), 
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
], additional_targets={"hsi_image": "image"})


file_data =  open(rgb_data_json)
file_contents = json.load(file_data)


id2label ={}
id2color = {}
for i, item in enumerate(file_contents['items'], start=0):
    id2label[i] = item['name']
    id2color[i] = [item['red_value'], item['green_value'], item['blue_value']]
    
# print(id2label)
# print(id2color)
num_classes = len(id2label)

train_dataset = LIBHSIDataset(image_set="train", root_dir=dataset_dir, id2color=id2color, transform=train_transform)
test_dataset = LIBHSIDataset(image_set="test", root_dir=dataset_dir, id2color=id2color,  transform=test_transform)
val_dataset = LIBHSIDataset(image_set="validation", root_dir=dataset_dir, id2color=id2color, transform=test_transform)

model = UNET_SemanticSegmentation(num_classes=num_classes,learning_rate=initial_lr, ignore_index=ignore_index, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, batch_size=batch_size)

checkpoint_callback_val_loss = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="lowest_val_loss_hsi")
checkpoint_callback_val_miou = ModelCheckpoint(monitor="val_miou", mode="max", save_top_k=1, filename="best_val_miou_hsi")
checkpoint_callback_last_epoch = ModelCheckpoint(monitor="epoch", mode="max", save_top_k=1, filename="last_epoch_hsi")

trainer = L.Trainer(max_epochs=max_num_epochs, accumulate_grad_batches=accumulate_grad_batches, callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=5), checkpoint_callback_val_loss,checkpoint_callback_last_epoch ,  checkpoint_callback_val_miou, StochasticWeightAveraging(swa_lrs=swa_lr) ], accelerator="gpu", devices="auto", gradient_clip_val=grad_clip_val, ) 

if training_model == True: 
    
    if tuning_model:
        tuner = Tuner(trainer)

        batch_finder = tuner.scale_batch_size(model, mode="binsearch")

        # below can be used to find the lr_for the model
        lr_finder = tuner.lr_find(model)

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        model.hparams.learning_rate = new_lr  # learning_rate
        model.hparams.batch_size = batch_finder

        print("learning rate:", model.hparams.learning_rate, "batch size:", model.hparams.batch_size)

        hparams = model.hparams
    else:
        model.hparams.learning_rate = initial_lr  # learning_rate
        model.hparams.batch_size = batch_size
        
    trainer.fit(model)

# model = UNET_SemanticSegmentation.load_from_checkpoint("lightning_logs/version_60/checkpoints/lowest_val_loss_hsi.ckpt")
model.eval()
# test the model on the test set
trainer.test(model)
