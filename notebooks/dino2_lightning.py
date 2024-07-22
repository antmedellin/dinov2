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

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for GUI-based environments
import matplotlib.pyplot as plt  

import seaborn as sns

# pip install lightning tensorboard torch-tb-profiler pandas matplotlib seaborn
# pip install --upgrade torchmetrics
# sudo apt-get install python3-tk

# tensorboard --logdir=./lightning_logs/
# ctrl shft p -> Python: Launch Tensorboard  select lightning logs



# color information for dataset
Color_Information = {
    "unknown_class": {"red_value": 255, "green_value": 0, "blue_value": 0, "class_value": 0},
    "grass": {"blue_value": 0, "green_value": 255, "red_value": 0, "class_value": 1},
    "vegetation": {"blue_value": 51, "green_value": 102, "red_value": 102, "class_value": 2},
    "trail": {"blue_value": 170, "green_value": 170, "red_value": 170, "class_value": 3},
    "sky": {"blue_value": 255, "green_value": 120, "red_value": 0, "class_value": 4},
    "object": {"blue_value": 0, "green_value": 0, "red_value": 0, "class_value": 5}
}

# Convert to Python dictionary
rgb2id = {(info["red_value"], info["green_value"], info["blue_value"]): info["class_value"] for info in Color_Information.values()}
id2rgb = {v: k for k, v in rgb2id.items()}


class FreiburgDataset(Dataset):
    def __init__(self, data_subset, root_dir, rgb2id, transform=None):
        
        self.data_subset = data_subset # train test or val
        self.transform = transform
        self.root = root_dir
        
        # assumes file structure is root/rgb and root/labels
        self.img_dir =  join(self.root, self.data_subset, "rgb")
        self.label_dir = join(self.root, self.data_subset, "GT_color")
        
        # get the number of images in the dataset
        self.num_images = len(os.listdir(self.img_dir))
        # make sure it equals the number of labels
        assert self.num_images == len(os.listdir(self.label_dir))
        
        self.img_labels = [f for f in os.listdir(self.label_dir)]
        
        self.rgb2id = rgb2id
        self.id2color_np = np.array(list(rgb2id.keys()))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_name, _ = os.path.splitext(self.img_labels[idx])
        
        img_name = img_name.split('_')[0]
        if self.data_subset == 'train':
            label_img_name = img_name + '_mask'
            rgb_img_name = img_name + '_Clipped'
        elif self.data_subset == 'test':
            label_img_name = img_name + '_mask'
            rgb_img_name = img_name + '_Clipped'
        
        
        label_path = None  # Initialize label_path to None
        
        try:
            if os.path.exists(os.path.join(self.label_dir, label_img_name+ '.png')):
                label_path = os.path.join(self.label_dir, label_img_name+ '.png')
            elif os.path.exists(os.path.join(self.label_dir, label_img_name+ '.jpg')):
                label_path = os.path.join(self.label_dir, label_img_name+ '.jpg')
            else:
                # print('label path does not exist for:', os.path.join(self.label_dir, label_img_name))
                return sys.exit()
                # pass
                
        except:
                label_img_name = img_name + '_Clipped'
                if os.path.exists(os.path.join(self.label_dir, label_img_name+ '.png')):
                    label_path = os.path.join(self.label_dir, label_img_name+ '.png')
                elif os.path.exists(os.path.join(self.label_dir, label_img_name+ '.jpg')):
                    label_path = os.path.join(self.label_dir, label_img_name+ '.jpg')
                else:
                    print('label path does not exist for:', os.path.join(self.label_dir, label_img_name))
                    return sys.exit()

        if label_path is None:
            raise ValueError(f"No label file found for image {img_name} in directory {self.label_dir}")


        if os.path.exists(os.path.join(self.img_dir, rgb_img_name+ '.jpg')):
            img_path = os.path.join(self.img_dir, rgb_img_name+ '.jpg')
        elif os.path.exists(os.path.join(self.img_dir, rgb_img_name+ '.png')):
            img_path = os.path.join(self.img_dir, rgb_img_name+ '.png')        
        else:
            print('img path does not exist for:', os.path.join(self.img_dir, rgb_img_name))
            return sys.exit()
            
        image = Image.open(img_path).convert('RGB')
        target = Image.open(label_path).convert('RGB')

        # Resize the image and target to make sure they are the same size
        # use nearest neighbor interpolation to preserve the label ids
        target = target.resize(image.size, Image.NEAREST)

        
        # Convert PIL Image to numpy array
        image = np.array(image)
        target = np.array(target)
        image = image.astype(np.float32)
        
        # result should be between 0 and 1
        image /= 255.0
        
        

        # # Prepare an empty array for the new target
        target_new = np.zeros(target.shape[:2], dtype=np.int32)


        
        
        # # Convert RGB to class id
        
        
        # Perform the mapping using numpy broadcasting and argmin
        for i, color in enumerate(self.id2color_np):
            # Find where in the target the current color is
            mask = np.all(target == color, axis=-1)
            
            # Wherever the color is found, set the corresponding index in target_new to the current class label
            target_new[mask] = i
        
        transformed = self.transform(image=image, mask=target_new)
        image, target_new = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])
        
        image = image.permute(2,0,1).float()
        
        return image, target_new
  
        
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
        
        def forward(self, pixel_values):
            assert not torch.isnan(pixel_values).any(), "NaN values in input pixel_values"            
            embeddings = self.dinov2.get_intermediate_layers(pixel_values)[0].squeeze()
            
            assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            embeddings = embeddings.reshape(-1, self.tokenW, self.tokenH, self.patch_descriptor_size)
            embeddings = embeddings.permute(0,3,1,2)
        
            assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
            
            logits = self.classifier(embeddings)
            # print( logits[0])
            assert not torch.isnan(logits).any(), "NaN values in logits"
            logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)
            
            return logits
        
        def log_cf(self, result_cf, step_type):
            
            confusion_matrix_computed = result_cf.detach().cpu().numpy()
            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize = (10,7))
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
                result_cf = self.val_confusion_matrix(preds, labels)
                result_miou = self.val_miou(preds, labels)
                result_acc_overall = self.val_acc_overall(preds, labels)
                results_acc_mean = self.val_acc_mean(preds, labels)
                self.log_cf(result_cf, step_type)
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
            
            pixel_values = batch["pixel_values"] 
            labels = batch["labels"]     
            
            logits = self.forward(pixel_values)
            loss = self.loss_fn(logits, labels) 
       
            self.log_data("train", logits, labels, loss)

            return loss
        
        def test_step(self, batch, batch_idx):
            
            pixel_values = batch["pixel_values"] 
            labels = batch["labels"]   
            
            logits = self.forward(pixel_values)
          
            loss = self.loss_fn(logits, labels) 
      
            self.log_data("test", logits, labels, loss)
            
            return loss
        
        def validation_step(self, batch, batch_idx):
            
            pixel_values = batch["pixel_values"] 
            labels = batch["labels"]             
            
            logits = self.forward(pixel_values)
            
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
        
   
def collate_fn(inputs):


    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)

    return batch


torch.cuda.empty_cache()


# these should be multiple of 14
img_height = 448
img_width = 448
test_transform = A.Compose([
    A.Resize(width=img_width, height=img_height), # dinov2 has a patch descriptor size for 14x14 pixels, so we need to resize the image to a multiple of 14. This will also affect the tokens. divide dimensions by 14 and set to dimensions of tokens, larger resolutions will lead to better performance
])


root_dir='/workspaces/dinov2/freiburg_forest_annotated'

train_dataset = FreiburgDataset( root_dir=root_dir, data_subset='train',transform=test_transform, rgb2id=rgb2id)

test_dataset = FreiburgDataset(root_dir=root_dir,data_subset='test', transform=test_transform, rgb2id=rgb2id)

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    total_vram = gpu_properties.total_memory / 1e9  # Convert bytes to GB
    print(f"Total VRAM on device: {total_vram:.2f} GB")
else:
    print("CUDA is not available. Check if GPU is available or if PyTorch is installed with CUDA.")


num_workers = 10 #  os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() is None


model = DinoV2SemanticSegmentation(num_classes=6, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=False, tokenW=img_width//14, tokenH=img_height//14, learning_rate=5e-5, ignore_index=0)

train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True, collate_fn=collate_fn,num_workers=10)
test_dataloader = DataLoader(test_dataset, batch_size=30, shuffle=False, collate_fn=collate_fn,num_workers=10)


checkpoint_callback_val_loss = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="lowest_val_loss")

checkpoint_callback_last_epoch = ModelCheckpoint(monitor="epoch", mode="max", save_top_k=1, filename="last_epoch")



trainer = L.Trainer(max_epochs=100, callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True), checkpoint_callback_val_loss,checkpoint_callback_last_epoch ])

trainer.fit(model, train_dataloader,  test_dataloader)


# # Load the model from a checkpoint
# model = DinoV2SemanticSegmentation.load_from_checkpoint("lightning_logs/version_7/checkpoints/epoch=81-step=656.ckpt")
# model.eval()

# # perform inference on a sample image 
# batch = dict()
# batch["pixel_values"] = train_dataset[0][0].unsqueeze(0)
# batch["labels"] = train_dataset[0][1].unsqueeze(0)

# test_img = batch["pixel_values"] #torch.as_tensor(batch["pixel_values"])
# # print(train_dataset[0])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# with torch.no_grad():
#     results = model(test_img.to(device))
# preds = torch.argmax(results, dim=1)


# cmap = 'viridis'

# plt.imshow(preds[0].cpu().numpy(), cmap=cmap)
# plt.colorbar()  
# plt.figure()
# plt.imshow(batch["labels"][0].cpu().numpy(), cmap=cmap)
# plt.colorbar()
# plt.show()