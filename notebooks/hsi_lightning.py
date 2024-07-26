# main.py
# ! pip install torchvision
import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L
import os
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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
            assert not torch.isnan(rgb_pixel_values).any(), "NaN values in input pixel_values"            
            embeddings = self.dinov2.get_intermediate_layers(rgb_pixel_values)[0].squeeze()
            
            assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            embeddings = embeddings.reshape(-1, self.tokenW, self.tokenH, self.patch_descriptor_size)
            embeddings = embeddings.permute(0,3,1,2)
        
            assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
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
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
            return {
            'optimizer': optimizer,
            "lr_scheduler": scheduler
             }
 
class UNET_SemanticSegmentation(L.LightningModule):
        def __init__(self, num_classes,repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=True , tokenW=64, tokenH=64, learning_rate = 1e-3, ignore_index=0 ,num_channels=204):
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
           
            
            self.tokenW = tokenW
            self.tokenH = tokenH
            self.learning_rate = learning_rate
            self.ignore_index = ignore_index
            
            
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
        
        
    
        def forward(self, hsi_pixel_values, rgb_pixel_values):
           

            # test out unet without rgb right now so just hyperspectral 
            concat_layers = []
            x = hsi_pixel_values
            
            # down layers 
            for down in self.double_conv_downs:
                # x_new = self.adjust_channels(hsi_pixel_values) 
                # print(1)
                x = down(x)
                # print(2)
                if down != self.double_conv_downs[-1]:
                    concat_layers.append(x)
                    x = self.max_pool_2x2(x)
        
            concat_layers = concat_layers[::-1]
            
            # up layers 
            for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
                x = up_trans(x)
                if x.shape != concat_layer.shape:
                    x = torchvision.transforms.functional.resize(x, concat_layer.shape[2:])
                
                concatenated = torch.cat((concat_layer, x), dim=1)
                x = double_conv_up(concatenated)
                
            x = self.final_conv(x)
            
            
            assert not torch.isnan(rgb_pixel_values).any(), "NaN values in input pixel_values"            
            embeddings = self.dinov2.get_intermediate_layers(rgb_pixel_values)[0].squeeze()
            
            assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            embeddings = embeddings.reshape(-1, self.tokenW, self.tokenH, self.patch_descriptor_size)
            embeddings = embeddings.permute(0,3,1,2)
        
            assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
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
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
            # return optimizer
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
            return {
            'optimizer': optimizer,
            "lr_scheduler": scheduler
             }       
   
def collate_fn(inputs):

    batch = dict()
    batch["hsi_pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["rgb_pixel_values"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[2] for i in inputs], dim=0).long()

    return batch


dataset_dir='/workspaces/LIB-HSI'
rgb_data_json = '/workspaces/dinov2/notebooks/lib_hsi_rgb.json'
batch_size = 2
ignore_index=-1
num_workers = 4 #  os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() is None
initial_lr = 0.0001 
# these should be multiple of 14 for dino model 
img_height = 448
img_width = 448


torch.cuda.empty_cache()

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    total_vram = gpu_properties.total_memory / 1e9  # Convert bytes to GB
    print(f"Total VRAM on device: {total_vram:.2f} GB")
else:
    print("CUDA is not available. Check if GPU is available or if PyTorch is installed with CUDA.")

test_transform = A.Compose([
    A.Resize(width=img_width, height=img_height), # dinov2 has a patch descriptor size for 14x14 pixels, so we need to resize the image to a multiple of 14. This will also affect the tokens. divide dimensions by 14 and set to dimensions of tokens, larger resolutions will lead to better performance
])


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
# print("num classes",num_classes)

train_dataset = LIBHSIDataset(image_set="train", root_dir=dataset_dir, id2color=id2color, transform=test_transform)

test_dataset = LIBHSIDataset(image_set="test", root_dir=dataset_dir, id2color=id2color,  transform=test_transform)

val_dataset = LIBHSIDataset(image_set="validation", root_dir=dataset_dir, id2color=id2color, transform=test_transform)


# model = DinoV2SemanticSegmentation(num_classes=num_classes, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=False, tokenW=img_width//14, tokenH=img_height//14, learning_rate=initial_lr, ignore_index=ignore_index)

model = UNET_SemanticSegmentation(num_classes=num_classes, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=False, tokenW=img_width//14, tokenH=img_height//14, learning_rate=initial_lr, ignore_index=ignore_index, num_channels=204)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=num_workers)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=num_workers)


checkpoint_callback_val_loss = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="lowest_val_loss_hsi")

checkpoint_callback_last_epoch = ModelCheckpoint(monitor="epoch", mode="max", save_top_k=1, filename="last_epoch_hsi")

trainer = L.Trainer(max_epochs=100, callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True), checkpoint_callback_val_loss,checkpoint_callback_last_epoch ])

# below trains the model 
# trainer.fit(model, train_dataloader,  val_dataloader)


# # Load the model from a checkpoint
# model = DinoV2SemanticSegmentation.load_from_checkpoint("lightning_logs/version_8/checkpoints/last_epoch_hsi.ckpt")
model = UNET_SemanticSegmentation.load_from_checkpoint("lightning_logs/version_35/checkpoints/lowest_val_loss_hsi.ckpt")
model.eval()
# # test the model on the test set
trainer.test(model, dataloaders=test_dataloader)



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