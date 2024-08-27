import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics.segmentation import MeanIoU
import torch.optim.lr_scheduler as lr_scheduler 
import matplotlib
# matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
import math
import random
import warnings
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

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

def collate_fn(inputs):

    batch = dict()
    batch["hsi_pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["rgb_pixel_values"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[2] for i in inputs], dim=0).long()

    return batch      
        
class BaseSegmentationModel(L.LightningModule):
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
            raise NotImplementedError("Subclasses should implement this method")
        
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
            
            logits, attn_weights_all = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss = self.loss_fn(logits, labels) 
            
            self.log_data(step_type, logits, labels, loss)

            return loss
        
        def test_step(self, batch, batch_idx):
            
            step_type = "test"
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            #added attn_weights_all for visualization of attention map of vision transformer models 
            logits, attn_weights_all = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss = self.loss_fn(logits, labels) 
            
            self.log_data(step_type, logits, labels, loss)

            return loss
        
        def validation_step(self, batch, batch_idx):
                
            step_type = "val"
            rgb_pixel_values = batch["rgb_pixel_values"]
            hsi_pixel_values = batch["hsi_pixel_values"]
            labels = batch["labels"]     
            
            logits, attn_weights_all = self.forward(hsi_pixel_values,rgb_pixel_values)
            loss = self.loss_fn(logits, labels) 
            
            self.log_data(step_type, logits, labels, loss)

            return loss
        
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

class UNET_SemanticSegmentation(BaseSegmentationModel):
    def __init__(self, num_classes, learning_rate=1e-3, ignore_index=0, num_channels=204, num_workers=4, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2):
        super().__init__(num_classes, learning_rate, ignore_index, num_channels, num_workers, train_dataset, val_dataset, test_dataset, batch_size)
        
        # can replace with models from segmentation_models_pytorch
        # refernce: https://segmentation-modelspytorch.readthedocs.io/en/latest/#models 
        self.hsi_unet = smp.Unet('resnet152', in_channels=self.num_channels, classes=self.num_classes, encoder_depth=5)

    def forward(self, hsi_pixel_values, rgb_pixel_values):
            
            x = self.hsi_unet(hsi_pixel_values)
            
            return x
        

class DINOv2_SemanticSegmentation(BaseSegmentationModel):
    def __init__(self, num_classes, learning_rate=1e-3, ignore_index=0, num_channels=204, num_workers=4, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg", half_precision=False , tokenW=32, tokenH=32 ):
        super().__init__(num_classes, learning_rate, ignore_index, num_channels, num_workers, train_dataset, val_dataset, test_dataset, batch_size)
        
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

    def forward(self, hsi_pixel_values, rgb_pixel_values):
            
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

# channel vit architecture here 
# base it on hcs_channel_vit.py 
# https://github.com/insitro/ChannelViT   

def visualize_attention_map(attn_weights, image, layer_idx=0, head_idx=0):
    """
    Visualize the attention map for a specific layer and head.
    
    Parameters:
    - attn_weights: List of attention weights from each layer.
    - image: The original input image.
    - layer_idx: Index of the layer to visualize.
    - head_idx: Index of the head to visualize.
    """
    attn = attn_weights[layer_idx][0, head_idx].detach().cpu().numpy()
    attn = attn.reshape(int(np.sqrt(attn.shape[0])), int(np.sqrt(attn.shape[0])))
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[0].set_title("Original Image")
    ax[1].imshow(attn, cmap='viridis')
    ax[1].set_title(f"Attention Map - Layer {layer_idx}, Head {head_idx}")
    plt.show()

# Function to visualize predictions and ground truth
def visualize_segmentation(predictions, ground_truth, image, num_classes):
    """
    Visualize the segmentation predictions and ground truth labels.
    
    Parameters:
    - predictions: The predicted segmentation map.
    - ground_truth: The ground truth segmentation map.
    - image: The original input image.
    """
    
    # Convert predictions to class labels
    predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
    ground_truth_labels = ground_truth.cpu().numpy()
    
    # Define a colormap and normalization
    cmap = ListedColormap(plt.cm.get_cmap('viridis', num_classes).colors)
    norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, num_classes)
    
    # Plot the original image, predicted labels, and ground truth labels
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[0].set_title("Original Image")
    ax[1].imshow(predicted_labels[0], cmap=cmap, norm=norm)
    ax[1].set_title("Predicted Labels")
    ax[2].imshow(ground_truth_labels[0], cmap=cmap, norm=norm)
    ax[2].set_title("Ground Truth Labels")
    plt.show()    
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.pos_embed

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    # def forward(self, x):
    #     x = x + self.self_attn(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))[0]
    #     x = x + self.mlp(self.layer_norm2(x))
    #     return x
    def forward(self, x):
        attn_output, attn_weights = self.self_attn(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))
        x = x + attn_output
        x = x + self.mlp(self.layer_norm2(x))
        return x, attn_weights

class ViT_SemanticSegmentation(BaseSegmentationModel):
    def __init__(self, num_classes, learning_rate=1e-3, ignore_index=0, num_channels=204, num_workers=4, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=2, patch_size = 16, embed_dim = 768, num_heads = 12, mlp_dim = 3072, num_layers = 12, dropout = 0.1):
        super().__init__(num_classes, learning_rate, ignore_index, num_channels, num_workers, train_dataset, val_dataset, test_dataset, batch_size)
        
        self.patch_embed = PatchEmbedding(num_channels, patch_size, embed_dim)
        # self.pos_embed = None  # Initialize positional encoding as None
        # self.pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size) ** 2, embed_dim))  # Initialize positional encoding
        
        # Calculate number of patches based on image dimensions and patch size
        num_patches = (448 // patch_size) ** 2  # Assuming input image size is 448x448
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))  # Initialize positional encoding
        
        
        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])
        # num_patches = (224 // patch_size) ** 2  # Assuming input image size is 224x224
        # self.pos_embed = PositionalEncoding(embed_dim, num_patches)
      
        self.segmentation_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, hsi_pixel_values, rgb_pixel_values):
        x = self.patch_embed(hsi_pixel_values)
        # Calculate number of patches dynamically
        B, N, C = x.shape
        # if self.pos_embed.shape[1] != N:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, N, C))
        self.pos_embed.data = self.pos_embed.data.to(x.device)        
        x = x + self.pos_embed
        attn_weights_all = []
        for blk in self.blocks:
            x, attn_weights = blk(x)
            attn_weights_all.append(attn_weights)
        
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.segmentation_head(x)
        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)  # Upsample to original image size
        return x, attn_weights_all