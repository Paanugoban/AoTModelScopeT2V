'''
Import statements for all the libraries we will be using in this script
'''
import torch
import os
import re
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import csv

'''
If you will be running on MAC, you can keep the following code, otherwise it needs to be changed to CUDA to run on GPU
'''
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    raise RuntimeError("MPS device not found.")

'''
This is the dataset for our RGB images that were extracted from videos that we will be using for training and testing.
Argument(s): 
    Root directory: 
        Type: String
        Description: Ensure you have all the images for either train or test the same directory. It should be in the format /root_dir/category/prompt/video_folder/frames,
        where category is the category of video, prompt is the prompt length, and video folder is the title of the video
    Transform:
        Type: torchvision.transforms 
        Description: This is the transform that will be applied to the images. Pass in the required transforms for train or test depending on the dataset. For train, transforms should
        nclude the data augmentations.
Return(s):
    Images, Labels, Video Title, Category, Prompt
'''
class RGBFramePredictionDataset_twice(Dataset):
    def __init__(self, root_dir, transform=None):
        #This is the root directory where all the images are stored
        self.root_dir = root_dir
        #This is the list of all the samples that will be used for training or testing
        self.samples = []
        #This is the transform that will be applied to the images
        self.transform = transform
        #Iterate through each category in the root directory
        for category in os.listdir(root_dir):
            #This is the path to the category
            category_path = os.path.join(root_dir, category)
            # Check if category_path is a directory
            if os.path.isdir(category_path):
                # Iterate through prompt folders '4','10','20','40','70'. You can adjust this if required
                for prompt in os.listdir(category_path):
                    # This is the path to the prompt folder
                    prompt_path = os.path.join(category_path, prompt)
                    # Check if prompt_path is a directory
                    if os.path.isdir(prompt_path):
                        # Iterate through video folders
                        video_folders = os.listdir(prompt_path)
                        # Check if video_path is a directory
                        for video_folder in video_folders: 
                            # This is the path to the video folder
                            video_path = os.path.join(prompt_path, video_folder)
                            # Check if video_path is a directory
                            if os.path.isdir(video_path):
                                # Filter out the files that are not images
                                filtered_files = [f for f in os.listdir(video_path) if re.match(r'[0-9]+\.png', f)]
                                # Sort the files in ascending order
                                sorted_files = sorted(filtered_files, key=lambda x: int(x.split('.')[0]))
                                image_files = [f for f in sorted_files if f.endswith('.png')]
                                # Check if there are 10 images in the folder
                                if len(image_files) == 10:
                                    # Append the sample to the list of samples
                                    self.samples.append((video_path, image_files, 1, category, prompt))
                                    # Append the sample to the list of samples again, but this time with the images in reverse order
                                    self.samples.append((video_path, image_files[::-1], 0, category, prompt))
    #This is required for the dataset to work, it returns the length of the dataset
    def __len__(self):
        return len(self.samples)
    #This is required for the dataset to work, it returns the sample at the index idx
    def __getitem__(self, idx):
        #This is the path to the video folder
        video_path, image_files, label, category, prompt = self.samples[idx]
        #Load the images from the video folder
        video_title = os.path.basename(video_path)
        #Load the images from the video folder
        images = [Image.open(os.path.join(video_path, image_file)).convert('RGB') for image_file in image_files]
        #Apply the sequences of transforms to the images
        if self.transform:
            images = [self.transform(img) for img in images]
        #Stack the images into a tensor
        image_sequence = torch.stack(images, axis=0)
        return image_sequence, label, video_title, category, prompt

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

'''
A function used to test different versions and architectures for rapid prototyping. The final mode 7 was what was final model. A sandbox with all modes will be made available in GitHub.
Parameter(s):
    test_batch_size:
        Type: Integer
        Description: Batch size for testing
    train_batch_size:
        Type: Integer
        Description: Batch size for training
    train_transform:
        Type: torchvision.transforms
        Description: Transforms to apply to the training images
    test_transform:
        Type: torchvision.transforms
        Description: Transforms to apply to the testing images
    type_model:
        Type: Integer
        Description: Current model used is 7, however in a sandbox that will be available on github, other modes will be shown.
        
Return(s):
    model:
        Type: torch.nn.Module
        Description: The model that will be used for training and testing
    train_dataloader:
        Type: torch.utils.data.DataLoader
        Description: The dataloader that will be used for training
    test_dataloader:
        Type: torch.utils.data.DataLoader
        Description: The dataloader that will be used for testing

'''
def trainer(test_batch_size,train_batch_size,train_transform=None,test_transform=None,type_model=1):
    #This is the dataset for our RGB images that were extracted from videos that we will be using for training and testing.
    dataset_train = RGBFramePredictionDataset_twice(root_dir='./Videos2/TrainVids', transform=train_transform)
    if len(dataset_train) == 0:
        raise RuntimeError('The training dataset is empty. Check your data sources and path.')
    #This is the dataset for our RGB images that were extracted from videos that we will be using for training and testing.
    train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=2)
    #This is the dataset for our RGB images that were extracted from videos that we will be using for training and testing.
    dataset_test = RGBFramePredictionDataset_twice(root_dir='./Videos2/TestVids', transform=test_transform)
    #This is the dataset for our RGB images that were extracted from videos that we will be using for training and testing.
    test_dataloader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=True, num_workers=2)
    #This is the model defintiion. It takes in the features extracted from the images and passes it through an LSTM layer. The output of the LSTM layer is our prediction.
    model = SequenceModel_RGB(num_classes=2).to(mps_device)  
    return model,train_dataloader,test_dataloader

if __name__ == '__main__':
    #Mode set to our final model
    mode=7
    
    #These are test transformations which makes limited changes besides changing to Tensor, resizing and normalizign per the pre-trained model's requirements
    transform_test = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #These are the training transformation. They incorporate the test transformations but also selective data augmentations to help avoid overfitting.
    transform_train = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)), 
    #transforms.RandomGrayscale(p=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    #With one model this is unnecessary, but this file was used to test different models and architectures. The final model was mode 7. Can be used to adapt for other experiments easily
    if mode==7:
        #This is the dataset for our RGB images that were extracted from videos that we will be using for training and testing.
        model,train_dataloader,test_dataloader=trainer(20,5,transform_train,transform_test,mode)
        #This will replace the model that was brough in from the trainer with the trained version of this model
        model=torch.load('model_path_fine_tuned_1.pth').to(mps_device)
        #This is the loss function that will be used for training
        criterion = nn.CrossEntropyLoss()

        #Hyperparameters
        max_lr = 0.0001
        epochs = 1
        #These are lists that will be used to create plots at the end of training
        lrs=[]
        train_losses=[]
        test_losses=[]
        #This is the optimizer that will be used for training
        optimizer = torch.optim.Adam(model.parameters(), max_lr,weight_decay=1e-4)
        #This is the learning rate scheduler that will be used for training
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dataloader))
        
        # Training loop
        for epoch in range(epochs):
            #This will set the model to training mode
            model.train()
            #These are lists that will be used to store the predictions and labels for the training set
            train_preds, train_labels,train_titles,train_categories,train_prompts = [], [],[],[],[]
            #This is the running loss for the training set
            running_loss = 0.0
            #Iterate through the training set
            for i, data in enumerate(tqdm(train_dataloader)):
                #Get the inputs and labels
                inputs, labels,title, category, prompt= data
                #Move the inputs and labels to the MPS device. If you are using CUDA this should be changed at the top of the script
                inputs, labels = inputs.float().to(mps_device), labels.to(mps_device)
                #Zero the parameter gradients
                optimizer.zero_grad()
                #Forward pass
                outputs = model(inputs)
                #Calculate the loss
                loss = criterion(outputs, labels)
                #Backward pass
                loss.backward()
                #Clip the gradients
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.2)
                #Update the parameters
                optimizer.step()
                #Update the learning rate
                scheduler.step()
                #Add the loss to the running loss
                running_loss += loss.item()
                #Get the predictions
                _, predicted = torch.max(outputs, 1)
                #Add the predictions and labels to the lists
                train_preds.extend(predicted.cpu().numpy())
                #Add the predictions and labels to the lists
                train_labels.extend(labels.cpu().numpy())
                #Add the predictions and labels to the lists
                train_titles.extend(title)
                #Add the predictions and labels to the lists
                train_categories.extend(category)
                #Add the predictions and labels to the lists
                train_prompts.extend(prompt)

            #Calculate the accuracy, f1, precision, and recall for the training set
            train_accuracy = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='macro')
            train_precision = precision_score(train_labels, train_preds, average='macro')
            train_recall = recall_score(train_labels, train_preds, average='macro')
            
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader)}, Train Accuracy: {train_accuracy}, Train F1: {train_f1}, Precision: {train_precision}, Recall: {train_recall}")

            #Write the results to a csv file. Ensure that the csv file is created with the correct headers before running this script
            with open('train_results2.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                for prompt,category,title, actual, predicted in zip(train_prompts,train_categories,train_titles, train_labels, train_preds):
                    writer.writerow([prompt,category,title, actual, predicted])

            #This will set the model to evaluation mode
            model.eval()
            #These are lists that will be used to store the predictions and labels for the testing set
            correct = 0
            total = 0
            test_preds, test_labels,test_titles,test_categories,test_prompts = [], [],[],[],[]
            #This is the running loss for the testing set
            val_loss = 0.0
            #Iterate through the testing set
            with torch.no_grad():
                #Iterate through the testing set
                for data in test_dataloader:
                    #Get the inputs and labels
                    inputs, labels, title,category,prompt= data
                    #Move the inputs and labels to the MPS device. If you are using CUDA this should be changed at the top of the script
                    inputs, labels = inputs.float().to(mps_device), labels.to(mps_device)
                    #Forward pass
                    outputs = model(inputs)
                    #Calculate the loss
                    loss = criterion(outputs, labels)
                    #Add the loss to the running loss
                    val_loss += loss.item()

                    #Get the predictions
                    _, predicted = torch.max(outputs, 1)
                    #Add the predictions and labels to the lists
                    test_preds.extend(predicted.cpu().numpy())
                    #Add the predictions and labels to the lists
                    test_labels.extend(labels.cpu().numpy())
                    #Add the predictions and labels to the lists
                    test_titles.extend(title)
                    #Add the predictions and labels to the lists
                    test_categories.extend(category)
                    #Add the predictions and labels to the lists
                    test_prompts.extend(prompt)

            #Calculate the accuracy, f1, precision, and recall for the testing set
            test_accuracy = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average='macro')
            test_precision = precision_score(test_labels, test_preds, average='macro')
            test_recall = recall_score(test_labels, test_preds, average='macro')

            print(f"Epoch {epoch+1}, Test Loss: {val_loss/len(test_dataloader)}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall}")

            #Write the results to a csv file. Ensure that the csv file is created with the correct headers before running this script
            with open('test_results2.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                for prompt,category,title, actual, predicted in zip(test_prompts,test_categories,test_titles, test_labels, test_preds):
                    writer.writerow([prompt,category,title, actual, predicted])

            #Append the losses and learning rates to the lists
            train_losses.append(running_loss/len(train_dataloader))
            lrs.append(optimizer.param_groups[0]['lr'])
            test_losses.append(val_loss / len(test_dataloader))
            torch.save(model, 'model_path_fine_tuned_2.pth')
        
        
        #Final Plot of LR and Lossess
        print('Finished Training')
        plt.plot(list(range(len(lrs))),lrs)
        plt.show()
        plt.cla()
        plt.clf()
        plt.plot(list(range(len(lrs))),lrs)
        ax=plt.axes()
        (p1,)=ax.plot(list(range(len(train_losses))),train_losses)
        p1.set_label("training accuracies")
        (p2,)=ax.plot(list(range(len(test_losses))),test_losses)
        p2.set_label("validation accuracies")
        ax.legend()
        plt.show()

    