





# let use fine tune maskformer on custom dataset lib hsi 



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
from transformers import MaskFormerForInstanceSegmentation
from transformers import MaskFormerImageProcessor


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

# def collate_fn(inputs):


#     batch = dict()
#     batch["hsi_pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
#     batch["rgb_pixel_values"] = torch.stack([i[1] for i in inputs], dim=0)
#     batch["labels"] = torch.stack([i[2] for i in inputs], dim=0)

#     return batch

# def collate_fn(batch):
#     inputs = list(zip(*batch))
#     images = inputs[0]
#     segmentation_maps = inputs[1]
#     # this function pads the inputs to the same size,
#     # and creates a pixel mask
#     # actually padding isn't required here since we are cropping
#     batch = preprocessor(
#         images,
#         segmentation_maps=segmentation_maps,
#         return_tensors="pt",
#     )

#     batch["original_images"] = inputs[2]
#     batch["original_segmentation_maps"] = inputs[3]
    
#     return batch


def collate_fn(inputs):


    # batch = dict()
    # batch["hsi_pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    images = torch.stack([i[1] for i in inputs], dim=0)
    segmentation_maps = torch.stack([i[2] for i in inputs], dim=0)
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    batch["original_images"] = images
    batch["original_segmentation_maps"] = segmentation_maps

    return batch

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
dataset_dir = "/workspaces/LIB-HSI"
rgb_data_json = '/workspaces/dinov2/notebooks/lib_hsi_rgb.json'
output_dir = "/workspaces/dinov2/output"


epochs = 50
num_warmup_epochs = 5
batch_size = 2
num_workers = 6
ignore_index=99

initial_lr = 0.00001  # Initial learning rate for warm-up
base_lr = 0.0001  # Learning rate after warm-up
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





test_dataset = SegmentationDataset(image_set="test", root_dir=dataset_dir, id2color=id2color, transform=base_transform)
train_dataset = SegmentationDataset(image_set="train", root_dir=dataset_dir, id2color=id2color, transform=base_transform)

val_dataset = SegmentationDataset(image_set="validation", root_dir=dataset_dir, id2color=id2color, transform=base_transform)

print("train dataset size: ", train_dataset.__len__())
print("test dataset size: ", test_dataset.__len__())
print("validation dataset size: ", val_dataset.__len__())






# Create a preprocessor
preprocessor = MaskFormerImageProcessor(ignore_index=99, do_reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)



train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)


batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,v[0].shape)
    
    

# Replace the head of the pre-trained model
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",   id2label=id2label,    ignore_mismatched_sizes=True)
    
outputs = model(batch["pixel_values"].float(),
                class_labels=batch["class_labels"],
                mask_labels=batch["mask_labels"]
                )

metric = evaluate.load("mean_iou")



device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device",device)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

running_loss = 0.0
num_samples = 0

highest_accuracy = 0
lowest_loss = 1000000
highest_iou = 0
highest_overall_accuracy = 0
epochs_since_improvement = 0
early_stop = False
# best_score = float('inf')
best_score = 0
current_loss = 0 
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Create a directory to save the best models
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# save model in sub directory named model
model_directory = os.path.join(output_dir, "model")
if not os.path.exists(model_directory):
    os.makedirs(model_directory)



for epoch in range(epochs):
    
    current_lr = update_learning_rate(optimizer, epoch, warmup_lr, base_lr, num_warmup_epochs)
    
    if epoch > num_warmup_epochs:
        scheduler.step(current_loss)
    print(f"Epoch: {epoch+1}, Learning Rate: {current_lr}")
    
    # print("Epoch:", epoch)
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].float().to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        # Backward propagation
        loss = outputs.loss
        loss.backward()

        batch_size = batch["pixel_values"].float().size(0)
        running_loss += loss.item()
        num_samples += batch_size

        # print (idx)
        # if idx % 5 == 0 :
        #     print("Loss:", running_loss/num_samples)
        #     break

        # Optimization
        optimizer.step()
        
        # if idx > 1 :
        #     break
        
        # pixel_values = batch["pixel_values"].float()
        
        # Forward pass
    model.eval()
    current_loss_accumulated = 0
    for idx, batch in enumerate(tqdm(test_dataloader)):
        
        
        pixel_values = batch["pixel_values"].float()
        with torch.no_grad():
            
            # outputs = model(pixel_values=pixel_values.to(device)
            # )
            
            outputs = model(
            pixel_values=batch["pixel_values"].float().to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
             )
            
            
            original_images = batch["original_images"]
            
            batch_size =  batch["pixel_values"].float().size(0)#2#outputs.shape[0] This fails on the last batch when it has 1 instead of 2
            
            # print(idx)
            # target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
            # print(outputs.loss)
            current_loss_accumulated += outputs.loss.item()
            
        target_sizes = [(448, 448) for _ in range(batch_size)]


        predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        ground_truth_segmentation_maps = batch["original_segmentation_maps"]
        metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
        
        
            
            # if idx > 5 :
            #     break
      
    # print("Mean IoU:", metric.compute(num_labels = len(id2label), ignore_index = 99)['mean_iou'])
    current_loss = current_loss_accumulated / len(test_dataloader)
    metrics = metric.compute(num_labels=len(id2label),
                                ignore_index=ignore_index,
                                reduce_labels=False,
    )
    print("Mean Accuracy:" ,metrics['mean_accuracy'])
    print("Overall Accuracy:" ,metrics['overall_accuracy'])
    print("Mean IoU:" ,metrics['mean_iou'])
    
    if metrics["mean_accuracy"] > highest_accuracy:
        highest_accuracy = metrics["mean_accuracy"]
        torch.save(model.state_dict(), os.path.join(model_directory, "maskformer_highest_accuracy_model.pth"))
        
    if current_loss < lowest_loss:
        lowest_loss = current_loss
        torch.save(model.state_dict(), os.path.join(model_directory, "maskformer_lowest_loss_model.pth"))
        
    if metrics["mean_iou"] > highest_iou:
        highest_iou = metrics["mean_iou"]
        torch.save(model.state_dict(), os.path.join(model_directory, "maskformer_highest_iou_model.pth"))
        
    if metrics["overall_accuracy"] > highest_overall_accuracy:
        highest_overall_accuracy = metrics["overall_accuracy"]
        torch.save(model.state_dict(), os.path.join(model_directory, "maskformer_highest_overall_accuracy_model.pth"))
    
    
    
    
    
    
    
    
# sudo apt-get install python3-tk
# sudo apt-get install libgdal-dev gdal-bin

# pip install GDAL












# from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
# from PIL import Image
# import requests
# url = "https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/ade20k.jpeg"
# url = "https://user-images.githubusercontent.com/590151/281234915-de8071bf-0e98-44be-ba9e-d9c9642c704f.jpg"

# image = Image.open(requests.get(url, stream=True).raw)

# # Loading a single model for all three tasks
# processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
# model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

# # Semantic Segmentation
# semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
# semantic_outputs = model(**semantic_inputs)
# # pass through image_processor for postprocessing
# # predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

# # Instance Segmentation
# instance_inputs = processor(images=image, task_inputs=["instance"], return_tensors="pt")
# instance_outputs = model(**instance_inputs)
# # pass through image_processor for postprocessing
# # predicted_instance_map = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]

# # Panoptic Segmentation
# panoptic_inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
# panoptic_outputs = model(**panoptic_inputs)
# # pass through image_processor for postprocessing
# # predicted_semantic_map = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]





# import torch

# example = semantic_inputs
# for k,v in example.items():
#   if isinstance(v, torch.Tensor):
#     print(k,v.shape)










# # import sys
# # sys.path.append('/workspaces/dinov2')
# # import dinov2.eval.segmentation_m2f.models.segmentors
# # import os
# import torch
# # import dinov2.eval.segmentation.utils.colormaps as colormaps
# # import numpy as np
# # from PIL import Image
# # import urllib
# # import mmcv
# # from mmcv.runner import load_checkpoint
# # from mmseg.apis import init_segmentor, inference_segmentor
# # import torch.nn as nn
# import matplotlib
# matplotlib.use('TkAgg')  # Use TkAgg for GUI-based environments
# import matplotlib.pyplot as plt  

# from transformers import Dinov2Model, Dinov2PreTrainedModel
# from transformers.modeling_outputs import SemanticSegmenterOutput

# checkpoint = torch.load("/workspaces/dinov2/output/model/dinov2_vitb14_ade20k_linear_head.pth")




# class LinearClassifier(torch.nn.Module):
#     def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
#         super(LinearClassifier, self).__init__()

#         self.in_channels = in_channels
#         self.width = tokenW
#         self.height = tokenH
#         self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

#     def forward(self, embeddings):
#         embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
#         embeddings = embeddings.permute(0,3,1,2)

#         return self.classifier(embeddings)

# class Dinov2ForSemanticSegmentation(torch.nn.Module):
#   def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg",   half_precision=False, device="cuda",  linear_head=None):
#     super().__init__()

#     self.device = device
    
#     self.dinov2 = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
    
#     #Dinov2Model(config)
#     self.classifier = linear_head#LinearClassifier(config.hidden_size, 32, 32, config.num_labels)

#   def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
#     # use frozen features
#     # outputs = self.dinov2(pixel_values,
#     #                         output_hidden_states=output_hidden_states,
#     #                         output_attentions=output_attentions)
    
    
#     # get the patch embeddings - so we exclude the CLS token
#     # patch_embeddings = outputs.last_hidden_state[:,1:,:]
#     # # use frozen features
#     # outputs = self.dinov2(pixel_values)
#     # # Assuming `outputs` is a namedtuple or similar structure where `last_hidden_state` can be accessed.
#     # # If `outputs` does not directly provide `last_hidden_state`, you'll need to adjust this line accordingly.
#     # patch_embeddings = outputs.last_hidden_state[:,1:,:]



#     # # convert to logits and upsample to the size of the pixel values
#     # logits = self.classifier(patch_embeddings)
#     # logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)
#     patch_embeddings = self.dinov2.get_intermediate_layers(pixel_values)[0].squeeze()
            
#     # convert to logits and upsample to the size of the pixel values
#     logits = self.classifier(patch_embeddings)
#     logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)
        
        

#     loss = None
#     if labels is not None:
#         # important: we're going to use 0 here as ignore index 
#         # as we don't want the model to learn to predict background
#         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
#         loss = loss_fct(logits.squeeze(), labels.squeeze())
    
#     return  SemanticSegmenterOutput(
#         loss=loss,
#         logits=logits
#     )
    
# class_head = LinearClassifier(768, 32, 32, 150)


# model = Dinov2ForSemanticSegmentation(repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14_reg",   half_precision=False, device="cuda", linear_head = class_head)
    
#     # "facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))

# # print(model)


# from PIL import Image
# import urllib

# def load_image_from_url(url: str) -> Image:
#     with urllib.request.urlopen(url) as f:
#         return Image.open(f).convert("RGB")

# pixel_values = load_image_from_url("https://dl.fbaipublicfiles.com/dinov2/images/example.jpg")


# from torchvision import transforms
# import torch

# # Assuming `pixel_values` is the Image object that needs to be transformed
# # Define the transformation
# transform = transforms.Compose([
#     transforms.Resize((448, 448)),  # Resize to the input size expected by the model
#     transforms.ToTensor(),  # Convert the image to a tensor
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to match the model's training
# ])

# # Apply the transformation
# pixel_values_tensor = transform(pixel_values)

# pixel_values_tensor = pixel_values_tensor.unsqueeze(0)


# model.to('cpu')
# model.eval()
# pixel_values_tensor = pixel_values_tensor.to('cpu')

# outputs = model(pixel_values_tensor)

# print(outputs)



# # -------------------------------------------


# # def load_config_from_url(url: str) -> str:
# #     with urllib.request.urlopen(url) as f:
# #         return f.read().decode()

# # def convert_sync_batchnorm_to_batchnorm(module):
# #     """
# #     Recursively convert all SyncBatchNorm layers in the model to BatchNorm layers.
# #     """
# #     if isinstance(module, nn.SyncBatchNorm):
# #         new_module = nn.BatchNorm2d(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
# #         if module.affine:
# #             with torch.no_grad():
# #                 new_module.weight = nn.Parameter(module.weight.clone().detach())
# #                 new_module.bias = nn.Parameter(module.bias.clone().detach())
# #                 new_module.running_mean = module.running_mean.clone().detach()
# #                 new_module.running_var = module.running_var.clone().detach()
# #         return new_module
# #     else:
# #         for name, child in module.named_children():
# #             module.add_module(name, convert_sync_batchnorm_to_batchnorm(child))
# #     return module


# # def render_segmentation(segmentation_logits, dataset):
# #     colormap = DATASET_COLORMAPS[dataset]
# #     colormap_array = np.array(colormap, dtype=np.uint8)
# #     segmentation_values = colormap_array[segmentation_logits + 1]
# #     return Image.fromarray(segmentation_values)

# # def load_image_from_url(url: str) -> Image:
# #     with urllib.request.urlopen(url) as f:
# #         return Image.open(f).convert("RGB")



# # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# # torch.cuda.empty_cache()
# # device = "cpu"
# # device = torch.device("cpu")
# # DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
# # CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
# # CHECKPOINT_URL = f"/workspaces/dinov2/output/model/dinov2_vitg14_ade20k_m2f.pth"
# # DATASET_COLORMAPS = {
# #     "ade20k": colormaps.ADE20K_COLORMAP,
# #     "voc2012": colormaps.VOC2012_COLORMAP,
# # }

# # EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"

    
# # cfg_str = load_config_from_url(CONFIG_URL)
# # cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

# # model = init_segmentor(cfg, device=device)
# # load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
# # # model.cuda()
# # model.eval()

# # # Convert SyncBatchNorm to BatchNorm
# # model = convert_sync_batchnorm_to_batchnorm(model)

# # image = load_image_from_url(EXAMPLE_IMAGE_URL)

# # array = np.array(image)[:, :, ::-1] # BGR
# # model.to('cpu')

# # segmentation_logits = inference_segmentor(model, array)[0]
# # segmented_image = render_segmentation(segmentation_logits, "ade20k")



# # print(model)

# # plt.imshow(segmented_image)
# # plt.show()




# # # pip install datasets --no-deps

# # # pip install -q datasets
# # # pip install requests==2.28.2
# # # pip install tqdm==4.65.0

# # # perform fine tuning on custom dataset 
# # from datasets import load_dataset
# # import numpy as np

# # import albumentations as A
# # from torch.utils.data import Dataset
# # import torch
# # from transformers import MaskFormerImageProcessor
# # import mmcv
# # from mmcv.runner import load_checkpoint
# # from mmseg.apis import init_segmentor, inference_segmentor
# # import urllib
# # import torch.nn as nn
# # import os
# # import sys
# # sys.path.append('/workspaces/dinov2')
# # import dinov2.eval.segmentation_m2f.models.segmentors # this line is need to get the model to load

# # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# # def convert_sync_batchnorm_to_batchnorm(module):
# #     """
# #     Recursively convert all SyncBatchNorm layers in the model to BatchNorm layers.
# #     """
# #     if isinstance(module, nn.SyncBatchNorm):
# #         new_module = nn.BatchNorm2d(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
# #         if module.affine:
# #             with torch.no_grad():
# #                 new_module.weight = nn.Parameter(module.weight.clone().detach())
# #                 new_module.bias = nn.Parameter(module.bias.clone().detach())
# #                 new_module.running_mean = module.running_mean.clone().detach()
# #                 new_module.running_var = module.running_var.clone().detach()
# #         return new_module
# #     else:
# #         for name, child in module.named_children():
# #             module.add_module(name, convert_sync_batchnorm_to_batchnorm(child))
# #     return module

# # def load_config_from_url(url: str) -> str:
# #     with urllib.request.urlopen(url) as f:
# #         return f.read().decode()
    
    

# # class SegmentationDataset(Dataset):
# #   def __init__(self, dataset, transform):
# #     self.dataset = dataset
# #     self.transform = transform

# #   def __len__(self):
# #     return len(self.dataset)

# #   def __getitem__(self, idx):
# #     item = self.dataset[idx]
# #     original_image = np.array(item["image"])
# #     original_segmentation_map = np.array(item["label"])

# #     transformed = self.transform(image=original_image, mask=original_segmentation_map)
# #     image, target = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])

# #     # convert to C, H, W
# #     image = image.permute(2,0,1)

# #     return image, target, original_image, original_segmentation_map


# # def visualize_map(image, segmentation_map):
    
# #     segmentation_map_np = np.asarray(segmentation_map)
# #     segmentation_map = segmentation_map_np
# #     color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
# #     for label, color in id2color.items():
# #         color_seg[segmentation_map == label, :] = color

# #     # Show image + mask
# #     img = np.array(image) * 0.5 + color_seg * 0.5
# #     img = img.astype(np.uint8)

# #     plt.figure(figsize=(15, 10))
# #     plt.imshow(img)
# #     plt.show()
    

# # dataset = load_dataset("EduardoPacheco/FoodSeg103")


# # dataset = dataset.shuffle(seed=1)
# # dataset = dataset["train"].train_test_split(test_size=0.2)
# # train_ds = dataset["train"]
# # test_ds = dataset["test"]

# # example = train_ds[0]
# # image = example["image"]
# # segmentation_map = example["label"]
     
# # # plt.figure()
# # # plt.imshow(segmentation_map)
# # # plt.figure()
# # # plt.imshow(image)
# # # plt.show()


# # id2label = {
# #     0: "background",
# #     1: "candy",
# #     2: "egg tart",
# #     3: "french fries",
# #     4: "chocolate",
# #     5: "biscuit",
# #     6: "popcorn",
# #     7: "pudding",
# #     8: "ice cream",
# #     9: "cheese butter",
# #     10: "cake",
# #     11: "wine",
# #     12: "milkshake",
# #     13: "coffee",
# #     14: "juice",
# #     15: "milk",
# #     16: "tea",
# #     17: "almond",
# #     18: "red beans",
# #     19: "cashew",
# #     20: "dried cranberries",
# #     21: "soy",
# #     22: "walnut",
# #     23: "peanut",
# #     24: "egg",
# #     25: "apple",
# #     26: "date",
# #     27: "apricot",
# #     28: "avocado",
# #     29: "banana",
# #     30: "strawberry",
# #     31: "cherry",
# #     32: "blueberry",
# #     33: "raspberry",
# #     34: "mango",
# #     35: "olives",
# #     36: "peach",
# #     37: "lemon",
# #     38: "pear",
# #     39: "fig",
# #     40: "pineapple",
# #     41: "grape",
# #     42: "kiwi",
# #     43: "melon",
# #     44: "orange",
# #     45: "watermelon",
# #     46: "steak",
# #     47: "pork",
# #     48: "chicken duck",
# #     49: "sausage",
# #     50: "fried meat",
# #     51: "lamb",
# #     52: "sauce",
# #     53: "crab",
# #     54: "fish",
# #     55: "shellfish",
# #     56: "shrimp",
# #     57: "soup",
# #     58: "bread",
# #     59: "corn",
# #     60: "hamburg",
# #     61: "pizza",
# #     62: "hanamaki baozi",
# #     63: "wonton dumplings",
# #     64: "pasta",
# #     65: "noodles",
# #     66: "rice",
# #     67: "pie",
# #     68: "tofu",
# #     69: "eggplant",
# #     70: "potato",
# #     71: "garlic",
# #     72: "cauliflower",
# #     73: "tomato",
# #     74: "kelp",
# #     75: "seaweed",
# #     76: "spring onion",
# #     77: "rape",
# #     78: "ginger",
# #     79: "okra",
# #     80: "lettuce",
# #     81: "pumpkin",
# #     82: "cucumber",
# #     83: "white radish",
# #     84: "carrot",
# #     85: "asparagus",
# #     86: "bamboo shoots",
# #     87: "broccoli",
# #     88: "celery stick",
# #     89: "cilantro mint",
# #     90: "snow peas",
# #     91: "cabbage",
# #     92: "bean sprouts",
# #     93: "onion",
# #     94: "pepper",
# #     95: "green beans",
# #     96: "French beans",
# #     97: "king oyster mushroom",
# #     98: "shiitake",
# #     99: "enoki mushroom",
# #     100: "oyster mushroom",
# #     101: "white button mushroom",
# #     102: "salad",
# #     103: "other ingredients"
# # }


# # id2color = {k: list(np.random.choice(range(256), size=3)) for k,v in id2label.items()}

    
# # # visualize_map(image, segmentation_map)


# # ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
# # ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# # train_transform = A.Compose([
# #     A.LongestMaxSize(max_size=1333),
# #     A.RandomCrop(width=512, height=512),
# #     A.HorizontalFlip(p=0.5),
# #     # A.Normalize(mean=ADE_MEAN, std=ADE_STD),
# # ])

# # test_transform = A.Compose([
# #     A.Resize(width=512, height=512),
# #     # A.Normalize(mean=ADE_MEAN, std=ADE_STD),

# # ])

# # train_dataset = SegmentationDataset(train_ds, transform=train_transform)
# # test_dataset = SegmentationDataset(test_ds, transform=test_transform)





# # # Create a preprocessor
# # preprocessor = MaskFormerImageProcessor(ignore_index=0, do_reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
     

# # from torch.utils.data import DataLoader

# # def collate_fn(batch):
# #     inputs = list(zip(*batch))
# #     images = inputs[0]
# #     segmentation_maps = inputs[1]
# #     # this function pads the inputs to the same size,
# #     # and creates a pixel mask
# #     # actually padding isn't required here since we are cropping
# #     batch = preprocessor(
# #         images,
# #         segmentation_maps=segmentation_maps,
# #         return_tensors="pt",
# #     )

# #     batch["original_images"] = inputs[2]
# #     batch["original_segmentation_maps"] = inputs[3]
    
# #     return batch

# # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
# # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
     


# # batch = next(iter(train_dataloader))
# # for k,v in batch.items():
# #   if isinstance(v, torch.Tensor):
# #     print(k,v.shape)
# #   else:
# #     print(k,v[0].shape)

# # labels = [id2label[label] for label in batch["class_labels"][0].tolist()]
# # print(labels)





# # # sys.exit()

# # DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
# # CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
# # CHECKPOINT_URL = f"/workspaces/dinov2/output/model/dinov2_vitg14_ade20k_m2f.pth"
# # device = torch.device("cpu")

# # cfg_str = load_config_from_url(CONFIG_URL)
# # cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
# # # print(cfg)

# # # print(cfg['train_cfg'])
# # # print(cfg.keys())

# # def find_key_in_nested_dict(d, target_key):
# #     if target_key in d:
# #         return d[target_key]
# #     for key, value in d.items():
# #         if isinstance(value, dict):
# #             result = find_key_in_nested_dict(value, target_key)
# #             if result is not None:
# #                 return result
# #     return None

# # # Usage
# # train_cfg = find_key_in_nested_dict(cfg, 'train_cfg')
# # # if train_cfg is not None:
# # #     print("Found 'train_cfg':", train_cfg)
# # # else:
# # #     print("'train_cfg' not found in cfg.")
    
    
    
# # # sys.exit()
# # model = init_segmentor(cfg, device=device)
# # load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
# # # model.cuda()
# # model.eval()

# # # Convert SyncBatchNorm to BatchNorm
# # model = convert_sync_batchnorm_to_batchnorm(model)



# # # batch = next(iter(train_dataloader))
# # # image = batch["pixel_values"]

# # # if not isinstance(image, torch.Tensor):
# # #     image = torch.tensor(image)
# # # if len(image.shape) == 3:  # If single image, add batch dimension
# # #     image = image.unsqueeze(0)

# # # # Convert image to float and normalize if necessary
# # # # Example normalization (adjust based on your model's requirements)
# # # image = image.float() / 255.0
# # from PIL import Image


# # def load_image_from_url(url: str) -> Image:
# #     with urllib.request.urlopen(url) as f:
# #         return Image.open(f).convert("RGB")


# # EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"

# # image = load_image_from_url(EXAMPLE_IMAGE_URL)
# # array = np.array(image)[:, :, ::-1] # BGR

# # model.to('cpu')

# # class ModelInputLogger:
# #     def __init__(self, model):
# #         self.model = model
    
# #     def __call__(self, *args, **kwargs):
# #         if args:  # Check if args is not empty
# #             print("Input shape:", args[0].shape)  # Assuming the first argument is the input tensor
# #         else:
# #             print("No positional arguments.")
# #         if kwargs:  # Check if kwargs is not empty
# #             print("Keyword arguments:", kwargs)
# #         return self.model(*args, **kwargs)
    
# #     def __getattr__(self, name):
# #         return getattr(self.model, name)

# # # Wrap your existing model with the logger
# # model = ModelInputLogger(model)


# # # segmentation_logits = inference_segmentor(model, array)[0]


# # # print(segmentation_logits.shape)


# # # Before converting the array to a tensor, make a copy to ensure it has a standard memory layout
# # array_copy = array.copy()

# # # Now proceed with the conversion using the copy
# # tensor = torch.tensor(array_copy).float()
# # tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Convert shape to 1 x C x H x W
# # tensor = tensor.double()  # Convert tensor to Double



# # # # 1. Convert array to a PyTorch tensor and add a batch dimension
# # # tensor = torch.tensor(array).float()
# # # tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Convert shape to 1 x C x H x W

# # # # 2. Normalize the tensor if necessary (example normalization values)
# # # mean = np.array([123.675, 116.28, 103.53])
# # # std = np.array([58.395, 57.12, 57.375])
# # # tensor = (tensor - mean[:, None, None]) / std[:, None, None]

# # # Ensure mean and std are of type Double
# # mean = np.array([123.675, 116.28, 103.53], dtype=np.float64)
# # std = np.array([58.395, 57.12, 57.375], dtype=np.float64)

# # # Convert tensor to Double before normalization and model inference
# # tensor = torch.tensor(array_copy).float().double()  # Convert tensor to Double immediately
# # tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Convert shape to 1 x C x H x W
# # tensor = (tensor - torch.tensor(mean[:, None, None], dtype=torch.float64)) / torch.tensor(std[:, None, None], dtype=torch.float64)


# # tensor = tensor.double()  # Convert tensor to Double


# # result = model(img=[tensor], return_loss=False, rescale=True, img_metas=[[{
# #     'filename': None,
# #     'ori_filename': None,
# #     'ori_shape': (480, 640, 3),
# #     'img_shape': (512, 704, 3),
# #     'pad_shape': (512, 704, 3),
# #     # 'scale_factor': array([1.0671875, 1.0666667, 1.0671875, 1.0666667], dtype=np.float32),
# #     'scale_factor': np.array([1.0671875, 1.0666667, 1.0671875, 1.0666667], dtype=np.float32),
# #     'flip': False,
# #     'flip_direction': 'horizontal',
# #     'img_norm_cfg': {
# #         'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
# #         'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
# #         'to_rgb': True
# #     }
# # }]])



# # batch = next(iter(train_dataloader))

# # img_metas = batch.get("img_metas", None)



# # image_data_example = {
# #     "img_shape": (512,512, 3),  # Example image shape (height, width, channels)
# #     "scale_factor": (1.0, 1.0),  # Example scale factor (scale_x, scale_y)
# #     "flip": False,  # Example flip status
# #     "filename": "example.jpg",  # Example filename
# #     "ori_shape": (512,512, 3),  # Original image shape
# #     "pad_shape": (512,512, 3),  # Image shape after padding
# #     "img_norm_cfg": {"mean": (123.675, 116.28, 103.53), "std": (58.395, 57.12, 57.375), "to_rgb": True},  # Example normalization config
# # }

# # # Assuming you have multiple images, you would repeat the above process for each image
# # # and then add the resulting dictionaries to the img_metas list
# # img_metas = [image_data_example]



# # # Assuming batch["class_labels"] contains the class indices for mask instances
# # class_indices = batch["class_labels"]

# # # If class_indices is a tensor, convert it to a list for easier manipulation
# # if isinstance(class_indices, torch.Tensor):
# #     class_indices = class_indices.tolist()

# # # Now, class_indices contains the class indices for each mask instance
# # print("Class Indices for Mask Instances:", class_indices)

# # # If you need to map these indices to class names using id2label (assuming id2label is a dictionary mapping indices to names)
# # class_indices = class_indices[0]  # Extract the tensor from the list if it's wrapped in a list

# # class_names = [id2label[index.item()] for index in class_indices]

# # # class_names = [id2label[index] for index in class_indices[0]]  # Adjust indexing based on your data structure
# # print("Class Names for Mask Instances:", class_names)






# # outputs = model(
# #                 batch["pixel_values"].float(),
# #                 # class_labels=batch["class_labels"],
# #                 # mask_labels=batch["mask_labels"],
# #                 img_metas=img_metas,
# #                 gt_semantic_seg=batch["class_labels"],
# #                 gt_labels=class_indices, # NEED TO FIX THIS
# #                 gt_masks=batch["mask_labels"],)




# # print(outputs)

# print("finished")