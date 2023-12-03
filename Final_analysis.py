#Import Statements
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from pytorch_i3d import InceptionI3d
import math
import csv
import shutil

'''
If you will be running on MAC, you can keep the following code, otherwise it needs to be changed to CUDA to run on GPU
'''
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    raise RuntimeError("MPS device not found.")
'''
This is the dataset class for the RGB frames. It is similar to the one used in final_train.py except the folder structure is different.
In this root directory we have forward and backward folders for each category. Each of these folders contains the video folders. They are already sorted correctly.
Multiple experiments were ran for project and file modified, for experiment with prompt, this will have to modified slightly to include prompt directory and prompt as a value returned
Argument(s):
    root_dir: The root directory of the dataset
    transform: The transforms to apply to each image
Returns:
    image_sequence, label, video_title, category_name
'''
class RGBFramePredictionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Initialize variables
        self.root_dir = root_dir
        self.samples = []
        self.transform = transform

        # Iterate through each category
        for category in os.listdir(root_dir):
            # Iterate through each video folder
            category_path = os.path.join(root_dir, category)
            # Check if the category_path is a directory
            if os.path.isdir(category_path):
                # Iterate through 'forward' and 'backward' folders
                for label_folder in ['backward', 'forward']:
                    # Iterate through each video folder
                    label_path = os.path.join(category_path, label_folder)
                    # Check if the label_path is a directory
                    if os.path.isdir(label_path):
                        # Iterate through each video folder
                        for video_folder in os.listdir(label_path):
                            # Iterate through each video folder
                            video_folder_path = os.path.join(label_path, video_folder)
                            # Check if the video_folder_path is a directory
                            if os.path.isdir(video_folder_path):
                                # Get all the image files in the video folder
                                image_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith('.png')])
                                if len(image_files) == 10:
                                    # Add the video folder path, image files, label, video folder, and category to the samples list
                                    label = 1 if label_folder == "forward" else 0
                                    self.samples.append((video_folder_path, image_files, label, video_folder, category))
    # Returns the length of the dataset
    def __len__(self):
        return len(self.samples)
    # Returns the image sequence, label, video title, and category name
    def __getitem__(self, idx):
        # Get the video folder path, image files, label, video title, and category name
        video_path, image_files, label, video_title, category_name = self.samples[idx]
        # Load each image in the image files list
        images = [Image.open(os.path.join(video_path, image_file)).convert('RGB') for image_file in image_files]
        # Apply the transforms to each image
        if self.transform:
            images = [self.transform(img) for img in images]
        # Convert the list of images to a tensor
        image_sequence = torch.stack(images, axis=0)
        return image_sequence, label, video_title, category_name

'''
This is the feature extractor for the RGB images. It uses a pre-trained ResNet18 model to extract features from the images. This is the 
first stage of our model. Each image will have its features extracted and then passed to the second stage of the model.
Argument(s):
    Forward method will get passed the image
Return(s):
    Forward method will return the features extracted from the image
'''
class FeatureExtractor_RGB(nn.Module):
    def __init__(self):
        super(FeatureExtractor_RGB, self).__init__()
        #Calling the pre-trained ResNet18 model
        resnet = torchvision.models.resnet18(pretrained=True)
        #Freezing all the layers in the model. It proved better than having it trained. You can try training it as well.
        for param in resnet.parameters():
            param.requires_grad = False
        #Removing the last layer (fully connected layer) to get features
        modules = list(resnet.children())[:-1]
        #Creating a sequential model with all the layers except the last layer
        self.resnet = nn.Sequential(*modules)
        #Creating a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        #Passing the image through the ResNet18 model
        x = self.resnet(x)
        #Passing the output of the ResNet18 model through the global average pooling layer
        x = self.global_avg_pool(x) 
        #Flatten the output for each image
        x = x.view(x.size(0), -1)
        return x
    
'''
This is the model defintiion. It takes in the features extracted from the images and passes it through an LSTM layer. The output of the LSTM layer is our prediction.
Argument(s):
    num_classes:
        Type: Integer
        Description: Number of classes in the dataset. For our dataset, it is 2 (forward and backward)
    forward method will get passed the images of a video
Return(s):
    Forward method will return the prediction
'''
class SequenceModel_RGB(nn.Module):
    def __init__(self, num_classes=2):
        super(SequenceModel_RGB, self).__init__()
        #Calling the feature extractor
        self.feature_extractor = FeatureExtractor_RGB()
        #Creating an LSTM layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        #Creating a fully connected layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        #Getting the batch size, sequence length, channels, height, and width of the input
        batch_size, sequence_length, C, H, W = x.size()
        #Reshaping the input to (batch_size*sequence_length, channels, height, width)
        x = x.view(batch_size * sequence_length, C, H, W)
        #Passing the input through the feature extractor
        x = self.feature_extractor(x)
        #Reshaping the input to (batch_size, sequence_length, -1)
        x = x.view(batch_size, sequence_length, -1)
        #Passing the input through the LSTM layer
        lstm_out, _ = self.lstm(x)
        #Passing the output of the LSTM layer through the fully connected layer
        x = self.fc(lstm_out[:, -1, :])
        return x


#Loading the model. This is critical for this file, as we are visualizing the images, meaning we are using trained model.
model=torch.load('model_path_fine_tuned_1.pth')
model=model.to(mps_device)
#We do this to allow for gradCAM, however we do not train model as we do not change the weights.
for param in model.parameters():
    param.requires_grad = True
for name, param in model.named_parameters():
    print(name, param.requires_grad)
    
#Creating the transforms for the images.
transform_valid = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize according to pre-trained models' requirements
    ])
#Creating and loading the dataset with the validation images
dataset_valid = RGBFramePredictionDataset(root_dir='./videos_finetuned', transform=transform_valid)
#This only works because each batch is 1 video, if we were to change the batch size, we would have to change the code.
#We are feeding 10 images per video to this validation loop.
valid_dataloader = DataLoader(dataset_valid, batch_size=1, shuffle=True, num_workers=0)

#Creating the lists to store the activations and gradients
activations = []
gradients = []

#Creating the hooks to store the activations and gradients
def get_activations_hook(module, input, output):
    activations.append(output)

def get_gradients_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

#Registering the hooks to the model
model.feature_extractor.resnet[7][1].conv2.register_backward_hook(get_gradients_hook)
model.feature_extractor.resnet[7][1].conv2.register_forward_hook(get_activations_hook)

#Creating the directory to store the images. If it already exists, we delete it and create a new one.
with open('predictions_finetuned.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Video Title', 'Category', 'Actual Label', 'Predicted Label'])

    #Iterating through the validation dataset
    for batch_idx, (input_sequence, labels, video_title, category_name) in enumerate(valid_dataloader):
        #Passing the input through the model
        input_sequence=input_sequence.to(mps_device)
        #Passing the labels through the model
        labels=labels.to(mps_device)
        #Passing the input through the model
        output = model(input_sequence)
        #Getting the predicted classes
        _, predicted_classes = output.max(dim=1)
        #Write results from each batch to csv file
        for title, category, actual, predicted in zip(video_title, category_name, labels, predicted_classes):
            csv_writer.writerow([title, category, actual.item(), predicted.item()])
        
        #Getting the loss
        predicted_scores = output[torch.arange(output.shape[0]), predicted_classes]
        #Getting the gradients
        predicted_scores.backward()
        grad = gradients[0]
        act = activations[0].detach()
        pooled_gradients = torch.mean(grad, dim=[0, 2, 3])
        #Getting the title and category
        title = video_title[0]
        category = category_name[0]
        
        #Getting the activations
        for i in range(512):
            act[:, i, :, :] *= pooled_gradients[i]
        #Getting the heatmap
        heatmap = torch.mean(act, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        super_imposed_imgs=[]
        
        # iterate over each item in the batch
        for j in range(heatmap.shape[0]):
            # Select the j-th image in the batch
            img = input_sequence[0][j].cpu().data
            print(img.shape)
            # Check if the tensor is already 2D (height x width); if not, we make it 2D
            if img.dim() == 4:
                # This squeezes out the batch dimension if it's 1
                img = img.squeeze(0)

            # Denormalize the image if necessary
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img.permute(1, 2, 0).numpy() * std + mean
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
        
            # Resize the heatmap to have the same size as the original image
            heatmap_norm = heatmap[j] / np.max(heatmap[j])
            heatmap_img = cv2.resize(heatmap_norm, (img.shape[1], img.shape[0]))
            heatmap_img = np.uint8(255 * heatmap_img)
            heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

            # Apply a threshold to the heatmap
            intensity_threshold = 0.2
            heatmap_img[heatmap_img < (intensity_threshold * 255)] = 0

            # Superimpose the heatmap on original image
            superimposed_img = heatmap_img * 0.4 + img
            super_imposed_imgs.append(superimposed_img)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # Adjust the figure size as needed

        # Plot the images
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        #Move the plot to a new commmon location
        output_dir = os.path.join("videos_finetuned", category, title)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # Create the directory
        os.makedirs(output_dir)
        for idx, img in enumerate(super_imposed_imgs):
            print(f"Title type: {type(title)}, Title value: {title}")
            print(f"Category type: {type(category)}, Category value: {category}")
            # If the directory exists, remove it
            img_path = os.path.join(output_dir, f"{title}_gradcam_{idx}.png")
            cv2.imwrite(img_path, img)
        #Clear the activations and gradients for the next image
        activations.clear()
        gradients.clear()