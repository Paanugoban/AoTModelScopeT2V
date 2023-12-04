#Import Statements
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
import os
from PIL import Image
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from pytorch_i3d import InceptionI3d
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import csv
import torch
#To run on GPU
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    raise RuntimeError("MPS device not found.")

'''
Random Crop Function for Optical Flow Dataset
Argument(s):
    img: image to be cropped
    crop_size: size of the crop
Return(s):
    Cropped image
'''
def random_crop(img, crop_size):
    y, x = img.shape[:2]
    start_x = np.random.randint(0, x - crop_size + 1)
    start_y = np.random.randint(0, y - crop_size + 1)
    return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

'''
Frame Prediction Dataset Class for images where optical flows are being extracted between two frames 
Argument(s):
    root_dir: root directory of the dataset
    transform: transform to be applied to the images
    max_samples: maximum number of samples to be loaded
Return(s): Flow Tensor, Label

'''
class FramePredictionDataset(Dataset):
    def __init__(self, root_dir, transform=None,max_samples=None,type_model=1):
        self.root_dir = root_dir
        self.samples = []
        self.max_samples_per_class = max_samples // 2
        self.type=type_model

        for label_folder in ["forward", "backward"]:
            label_path = os.path.join(root_dir, label_folder)
            videos = os.listdir(label_path)
            sample_count = 0

            for video in videos:
                if sample_count >= self.max_samples_per_class:
                    break
                video_path = os.path.join(label_path, video)
                if os.path.isdir(video_path):
                    flow_files_x = sorted([f for f in os.listdir(video_path) if f.endswith('_x.png')])
                    flow_files_y = sorted([f for f in os.listdir(video_path) if f.endswith('_y.png')])
                    
                    if len(flow_files_x) == 9 and len(flow_files_y) == 9:
                        label = 1 if label_folder == "forward" else 0
                        self.samples.append((video_path, flow_files_x, flow_files_y, label))
                        sample_count += 1
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, flow_files_x, flow_files_y, label = self.samples[idx]

        crop_size = 28  # Adjust this size as desired
        flows_x = [cv2.resize(random_crop(cv2.imread(os.path.join(video_path, flow_file), cv2.IMREAD_GRAYSCALE), crop_size), (32, 32)) for flow_file in flow_files_x]
        flows_y = [cv2.resize(random_crop(cv2.imread(os.path.join(video_path, flow_file), cv2.IMREAD_GRAYSCALE), crop_size), (32, 32)) for flow_file in flow_files_y]

        # Normalize flows
        normalized_flows = []
        for fx, fy in zip(flows_x, flows_y):
            #fx_norm = (fx - fx.min()) / (fx.max() - fx.min())
            fx_norm=fx/255.0#make sure fx isn't an integer
            fy_norm=fy/255.0
            #fy_norm = (fy - fy.min()) / (fy.max() - fy.min())    
            normalized_flows.append(np.stack((fx_norm, fy_norm), axis=2))
        
        if self.type == 1:
            concatenated_flows = np.concatenate(normalized_flows, axis=2)

            # Random horizontal flip
            if np.random.rand() < 0.5:
                concatenated_flows = np.flip(concatenated_flows, axis=1)

            concatenated_flows = np.ascontiguousarray(concatenated_flows)
            concatenated_flows_tensor = torchvision.transforms.functional.to_tensor(concatenated_flows)

            return concatenated_flows_tensor, label

        else:  # self.type == 2
            flows_sequence = np.stack(normalized_flows, axis=0)

            # Random horizontal flip
            if np.random.rand() < 0.5:
                flows_sequence = np.flip(flows_sequence, axis=2)

            flows_sequence = np.ascontiguousarray(flows_sequence)
            flows_sequence_tensor = torch.as_tensor(flows_sequence.transpose(0, 3, 1, 2), dtype=torch.float32)

            return flows_sequence_tensor, label

'''
This dataset takes RGB frames as input from the sequence, however it only takes each video in one direction.
Argument(s):
    root_dir: root directory of the dataset
    transform: transform to be applied to the images
    max_samples: maximum number of samples to be loaded
Return(s): Image Tensor, Label
'''
class RGBFramePredictionDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=500,flow=False):
        self.root_dir = root_dir
        self.samples = []
        self.flow=flow
        self.max_samples_per_class = max_samples // 2
        self.transform = transform

        if self.flow:
            for label_folder in ["forward", "backward"]:
                label_path = os.path.join(root_dir, label_folder)
                videos = os.listdir(label_path)
                sample_count = 0
                for video in videos:
                    if sample_count >= self.max_samples_per_class:
                        break
                    video_path = os.path.join(label_path, video)
                    if os.path.isdir(video_path):
                        image_files = sorted([f for f in os.listdir(video_path) if f.endswith('.png')])
                        if len(image_files) == 10:
                            self.samples.append((video_path, image_files, 1 if label_folder == "forward" else 0))
        else:
            for category in os.listdir(root_dir):
                category_path=os.path.join(root_dir,category)
                for prompt in os.listdir(category_path):
                    prompt_path=os.path.join(category_path,prompt)
                    video_folders = os.listdir(prompt_path)

                    # Assign forward label to first half
                    for video_folder in video_folders: 
                        video_path = os.path.join(root_dir, video_folder)
                        if os.path.isdir(video_path):
                            image_files = sorted([f for f in os.listdir(video_path) if f.endswith('.png')])
                            if len(image_files) == 10:
                                self.samples.append((video_path, image_files, 1,category,prompt))
                                self.samples.append((video_path, image_files, 0,category,prompt)) 
                                # Assign reverse label to second half
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, image_files, label,category,prompt = self.samples[idx]
        video_title = os.path.basename(video_path)
        
        images = [Image.open(os.path.join(video_path, image_file)).convert('RGB') for image_file in image_files]
        
        # If the label is 1 (forward), do nothing. If the label is 0 (backward), reverse the sequence
        if label == 0 and self.flow==False:
            images = images[::-1]
        
        if self.transform:
            images = [self.transform(img) for img in images]

        image_sequence = torch.stack(images, axis=0)

        return image_sequence, label,video_title,category,prompt

 '''
 RGBFramePredictionDataset_twice is a dataset that takes RGB frames as input from the sequence, however it takes each video in both directions.
 It is what ultimately was chosen to be used for the final model.
 Argument(s):
    root_dir: root directory of the dataset
    transform: transform to be applied to the images
    max_samples: maximum number of samples to be loaded
Return(s): Image Tensor, Label
 '''
class RGBFramePredictionDataset_twice(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=500, flow=False):
        self.root_dir = root_dir
        self.samples = []
        self.flow = flow
        self.max_samples = max_samples
        self.transform = transform

        if self.flow:
            # Previous logic for when flow=True remains unchanged
            for label_folder in ["forward", "backward"]:
                label_path = os.path.join(root_dir, label_folder)
                videos = os.listdir(label_path)
                sample_count = 0
                for video in videos:
                    if sample_count >= self.max_samples // 2:
                        break
                    video_path = os.path.join(label_path, video)
                    if os.path.isdir(video_path):
                        filtered_files = [f for f in os.listdir(video_path) if re.match(r'.*_[0-9]+\.png', f)]
                        sorted_files = sorted(filtered_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                        image_files = [f for f in sorted_files if f.endswith('.png')]
                        if len(image_files) == 10:
                            self.samples.append((video_path, image_files, 1 if label_folder == "forward" else 0))
                            sample_count += 1
        else:
            for category in os.listdir(root_dir):
                category_path = os.path.join(root_dir, category)
                # Check if category_path is a directory
                if os.path.isdir(category_path):
                    for prompt in os.listdir(category_path):
                        prompt_path = os.path.join(category_path, prompt)
                        # Check if prompt_path is a directory
                        if os.path.isdir(prompt_path):
                            video_folders = os.listdir(prompt_path)

                            # Assign forward label to first half
                            for video_folder in video_folders: 
                                video_path = os.path.join(prompt_path, video_folder) # Fixed path
                                # Check if video_path is a directory
                                if os.path.isdir(video_path):
                                    filtered_files = [f for f in os.listdir(video_path) if re.match(r'[0-9]+\.png', f)]
                                    sorted_files = sorted(filtered_files, key=lambda x: int(x.split('.')[0]))
                                    image_files = [f for f in sorted_files if f.endswith('.png')]
                                    if len(image_files) == 10:
                                        self.samples.append((video_path, image_files, 1, category, prompt))
                                        self.samples.append((video_path, image_files[::-1], 0, category, prompt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, image_files, label, category, prompt = self.samples[idx]
        video_title = os.path.basename(video_path)
        images = [Image.open(os.path.join(video_path, image_file)).convert('RGB') for image_file in image_files]
        
        if self.transform:
            images = [self.transform(img) for img in images]

        image_sequence = torch.stack(images, axis=0)

        return image_sequence, label, video_title, category, prompt


'''
Dataset used in conjunction with Transformer based model
'''
class RGBFramePredictionDatasetSubject(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=100):
        self.root_dir = root_dir
        self.samples = []
        self.transform = transform

        # List only the directories in the root_dir
        image_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        # If there are fewer image folders than max_samples, just use all of them
        sampled_folders = image_folders[:max_samples]

        for i, image_folder in enumerate(sampled_folders):
            image_folder_path = os.path.join(root_dir, image_folder)

            # List the frames (assuming they're PNG images) in the folder and sort them
            frame_files = sorted([frame for frame in os.listdir(image_folder_path) if frame.endswith('.png')])

            # Instead of loading images here, just store their paths and the corresponding folder path
            label = 1 if i < len(sampled_folders) // 2 else 0
            self.samples.append((image_folder_path, frame_files, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, image_files, label = self.samples[idx]

        # Load images on-the-fly
        images = [Image.open(os.path.join(video_path, image_file)).convert('RGB') for image_file in image_files]

        # If label is 0 (reverse), reverse the sequence of images
        if label == 0:
            images = images[::-1]

        if self.transform:
            images = [self.transform(img) for img in images]

        image_sequence = torch.stack(images, axis=0)

        return image_sequence, label


'''
Dataset in Use with 3D convolutional nets
'''
class DualStreamDataset(Dataset):
    
    def __init__(self, root=None, transforms=None,max_samples=None):
        if not root:
            root = os.getcwd()
        self.rgb_root = os.path.join(root, 'trainFrames')
        self.flow_root = os.path.join(root, 'trainVideos1')
        self.transforms = transforms

        def filter_dirs(files):
            return [f for f in files if os.path.isdir(os.path.join(self.rgb_root, f)) and f != '.DS_Store']

        # List of all video folders for RGB frames
        self.rgb_video_list = sorted(filter_dirs(os.listdir(self.rgb_root)))

        # List of all video folders for optical flows in forward and backward directions
        self.forward_flow_list = sorted(filter_dirs(os.listdir(os.path.join(self.flow_root, 'forward'))))
        self.backward_flow_list = sorted(filter_dirs(os.listdir(os.path.join(self.flow_root, 'backward'))))

        if max_samples:
            half_samples = max_samples // 2
            self.forward_flow_list = self.forward_flow_list[:half_samples]
            self.backward_flow_list = self.backward_flow_list[:half_samples]

        # Create a unified list for flow videos along with their direction (for label assignment)
        self.flow_video_list = [(v, 'forward') for v in self.forward_flow_list] + [(v, 'backward') for v in self.backward_flow_list]

    def __getitem__(self, index):
        vid, direction = self.flow_video_list[index]

        # Load RGB frames
        rgb_images = []
        for i in range(9):
            img_path = os.path.join(self.rgb_root, vid, f'{i}.png')
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0  # Convert BGR to RGB
            if self.transforms:
                img = self.transforms(img)
            rgb_images.append(torch.tensor(img))

        # If direction is backward, reverse the order of RGB images
        if direction == 'backward':
            rgb_images = rgb_images[::-1]

        rgb_images = torch.stack(rgb_images).permute(3, 0, 1, 2)

        # Load Optical Flows
        flow_images = []
        for i in range(9):
            flow_x_path = os.path.join(self.flow_root, direction, vid, f'{i}_x.png')
            flow_y_path = os.path.join(self.flow_root, direction, vid, f'{i}_y.png')
            
            flow_x = cv2.imread(flow_x_path, cv2.IMREAD_GRAYSCALE)/255.0
            flow_y = cv2.imread(flow_y_path, cv2.IMREAD_GRAYSCALE)/255.0

            if self.transforms:
                flow_x = self.transforms(flow_x)
                flow_y = self.transforms(flow_y)
                
            flow_images.append(torch.stack([torch.tensor(flow_x), torch.tensor(flow_y)]))
        flow_images = torch.stack(flow_images).permute(1, 0, 3, 2)
        #print(flow_images.size())

        # Label assignment (0 for forward, 1 for backward)
        label = 0 if direction == 'forward' else 1

        return rgb_images, flow_images, label

    def __len__(self):
        return len(self.flow_video_list)

'''
2D convolutional Classifier as Test of Concept from literature
'''
class CustomClassifier(nn.Module):
    def __init__(self, num_classes=2, num_optical_flows=9):
        super(CustomClassifier, self).__init__()
        
        # Load the model (not pretrained)
        self.vgg16 = torchvision.models.vgg16()
        #self.resnet = torchvision.models.resnet50()
        
        # Modify the first convolutional layer to accept num_optical_flows * 2 channels
        self.vgg16.features[0] = nn.Conv2d(num_optical_flows * 2, 64, kernel_size=3, padding=1)
        
        # Custom classifier layers
        self.classifier = nn.Sequential(
            #nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        # Use VGG's feature extraction
        x = self.vgg16.features(x)
        #x = self.features(x)
        # Custom classifier
        x = self.classifier(x)
        return x

'''
Implementatino of Resnet 9 as an alternative VGG16
'''
class ResNet9(nn.Module):
    def __init__(self, in_channels=2, num_classes=128):
        super(ResNet9, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bnorm1=nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bnorm2=nn.BatchNorm2d(128)
        self.res1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(128))
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bnorm3=nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bnorm4=nn.BatchNorm2d(512)
        self.res2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(),
                                  nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(512))
        
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = x + self.res1(x)
        x = F.relu(x)
        
        x = F.relu(self.bnorm3(self.conv3(x)))
        x = F.relu(self.bnorm4(self.conv4(x)))
        x = x + self.res2(x)
        x = F.relu(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

'''
Original Feature Extractor to be used with Resnet 9 and optical flows
'''
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = ResNet9()

    def forward(self, x):
        x=self.resnet(x)
        return x

'''
Original Sequence model to be used with optical flows and LSTM to see if prediction can be made on final direciton
'''
class SequenceModel(nn.Module):
    def __init__(self, num_classes=2):  # Assuming 10 classes, adjust as necessary
        super(SequenceModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, sequence_length, C, H, W = x.size()
        
        # Process each image in the sequence with CNN
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.feature_extractor(x)
        
        # Restore the sequence structure
        x = x.view(batch_size, sequence_length, -1)
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output of the LSTM sequence for classification
        x = self.fc(lstm_out[:, -1, :])
        return x

'''
Final Model selected for feature extractor from frames, please see clean implementation in final_train file
'''
class FeatureExtractor_RGB(nn.Module):
    def __init__(self):
        super(FeatureExtractor_RGB, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        # Unfreeze the last layer (it's the layer before the final fully connected layer in ResNet18)
        # for param in resnet.layer4.parameters():
        #     param.requires_grad = True
        # Removing the last layer (fully connected layer) to get features
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.resnet(x)
        x = self.global_avg_pool(x) 
        # Flatten the output for each image
        x = x.view(x.size(0), -1)
        return x

'''
Final Model selected. Please see cleaned versin in final_train.py
'''
class SequenceModel_RGB(nn.Module):
    def __init__(self, num_classes=2):
        super(SequenceModel_RGB, self).__init__()
        self.feature_extractor = FeatureExtractor_RGB()
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.feature_extractor(x)
        x = x.view(batch_size, sequence_length, -1)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x

'''
Class for positional Encoding made to be used with Transformer Based Model
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10):  # Changed max_len to 10
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

'''
Transformer block to be used in conjunction with Sequence Model_RGB_transformer
'''
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers=1):
        super(TransformerBlock, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src):
        output = self.transformer_encoder(src)
        return output

'''
This is the final model selected for the project. It is a transformer based model that takes in RGB frames as input.
'''
class SequenceModel_RGB_Transformer(nn.Module):
    def __init__(self, num_classes=2):
        super(SequenceModel_RGB_Transformer, self).__init__()
        self.feature_extractor = FeatureExtractor_RGB()
        
        # For the transformer, we'll need to determine the d_model (dimension of model)
        # Assuming the output of the feature extractor is 512, we'll use that.
        self.d_model = 512
        self.nhead = 8  # number of heads in multiheadattention
        self.transformer = TransformerBlock(d_model=self.d_model, nhead=self.nhead)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        # After the transformer, we'll use the last output for classification.
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.feature_extractor(x)
        
        # Transformer expects input in shape (S, N, E) where S is source sequence length, 
        # N is batch size, and E is feature dimension.
        # So we reshape the output from the feature extractor accordingly.
        x = x.view(sequence_length, batch_size, self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Use the last output for classification
        x = self.fc(x[-1])
        return x

'''
Trainer having various modes of operation helping for prototyping and selecting the ideal model
'''
def trainer(test_batch_size,train_batch_size,train_samples,test_samples,train_transform=None,test_transform=None,type_model=1):
    print(f"Inside trainer with type_model: {type_model}")
    if type_model==1:
        dataset_train = FramePredictionDataset(root_dir='./trainVideos1', transform=train_transform,max_samples=train_samples)
        train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=2)
        dataset_test = FramePredictionDataset(root_dir='./testVideos1', transform=test_transform,max_samples=test_samples)#1200
        test_dataloader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=True, num_workers=2)
        model = CustomClassifier(num_classes=2).to(mps_device) 
        return model,train_dataloader,test_dataloader 
    elif type_model==2:
        dataset_train = FramePredictionDataset(root_dir='./trainVideos1', transform=train_transform,max_samples=train_samples,type_model=2)
        train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=2)
        dataset_test = FramePredictionDataset(root_dir='./testVideos1', transform=test_transform,max_samples=test_samples,type_model=2)#1200
        test_dataloader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=True, num_workers=2)
        model = SequenceModel(num_classes=2).to(mps_device) 
        return model,train_dataloader,test_dataloader 
    elif type_model==3:
        dataset_train = RGBFramePredictionDataset(root_dir='./trainFrames', transform=train_transform,max_samples=train_samples,flow=False)
        train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=2)
        dataset_test = RGBFramePredictionDataset(root_dir='./testFrames', transform=test_transform,max_samples=test_samples,flow=False)#1200
        test_dataloader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=True, num_workers=2)
        model = SequenceModel_RGB(num_classes=2).to(mps_device)  
        return model,train_dataloader,test_dataloader
    elif type_model==4:
        dataset_train = RGBFramePredictionDataset(root_dir='./trainFrames', transform=train_transform,max_samples=train_samples)
        train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=2)
        dataset_test = RGBFramePredictionDataset(root_dir='./testFrames', transform=test_transform,max_samples=test_samples)#1200
        test_dataloader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=True, num_workers=2)
        model = SequenceModel_RGB_Transformer(num_classes=2).to(mps_device)
        return model,train_dataloader,test_dataloader
    elif type_model==5:
        train_dataset = DualStreamDataset(transforms=train_transform,max_samples=2000)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
        i3d_rgb = InceptionI3d(400, in_channels=3)  # For RGB streams
        i3d_flow = InceptionI3d(400, in_channels=2)  # For Flow streams
        return i3d_rgb,i3d_flow,train_loader
    elif type_model==6:
        dataset_train = RGBFramePredictionDatasetSubject(root_dir='./cleanedRunning', transform=train_transform,max_samples=train_samples)
        train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=0)
        dataset_test = RGBFramePredictionDatasetSubject(root_dir='./cleanedRunningTest', transform=test_transform,max_samples=test_samples)#1200
        test_dataloader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=True, num_workers=0)
        model = SequenceModel_RGB_Transformer(num_classes=2).to(mps_device)  
        return model,train_dataloader,test_dataloader
    else:
        dataset_train = RGBFramePredictionDataset_twice(root_dir='./Videos2/TrainVids', transform=train_transform,max_samples=train_samples,flow=False)
        if len(dataset_train) == 0:
            raise RuntimeError('The training dataset is empty. Check your data sources and path.')
        train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=2)
        dataset_test = RGBFramePredictionDataset_twice(root_dir='./Videos2/TestVids', transform=test_transform,max_samples=test_samples,flow=False)#1200
        test_dataloader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=True, num_workers=2)
        model = SequenceModel_RGB(num_classes=2).to(mps_device)  
        return model,train_dataloader,test_dataloader


def visualize_batch(batch_data, batch_labels, num_images=10):
    """Visualizes the first num_images from the batch."""
    # Select the first video (sequence of images) from the batch
    first_video_frames = batch_data[0]
    first_video_label = batch_labels[0]

    # Display num_images frames from the selected video
    for i, frame in enumerate(first_video_frames[:num_images]):
        plt.subplot(1, 10, i+1)  # Assuming grid is 2x5 for 10 images
        plt.imshow(frame.permute(1, 2, 0).cpu().numpy())  # Make sure to move tensor to cpu
        plt.title(f"Label: {first_video_label.item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    #mps.empty_cache
    mode=7
    
    transform_test = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize according to pre-trained models' requirements
    ])

    # transform_test = transforms.Compose([
    # transforms.ToTensor(), # Normalize according to pre-trained models' requirements
    # ])

    transform_train = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)), 
    #transforms.RandomGrayscale(p=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize according to pre-trained models' requirements
    ])

    # transform_train = transforms.Compose([
    # transforms.ToTensor(),  # Normalize according to pre-trained models' requirements
    # ])

    #The 3D convolution method is soo different it can't share the same method of training as other methods
    if mode==5:
        i3d_rgb,i3d_flow,train_loader=trainer(4,32,2000,400,type_model=5)

        i3d_rgb
        i3d_flow
        optimizer = torch.optim.Adam(list(i3d_rgb.parameters()) + list(i3d_flow.parameters()), lr=0.001)

        criterion = nn.CrossEntropyLoss()
        epochs=10
        # Training loop
        for epoch in range(epochs):
            for rgb_data, flow_data, labels in tqdm(train_loader):
                rgb_data = rgb_data.float()
                rgb_data = rgb_data
                flow_data = flow_data.to(dtype=torch.float32)
                labels = labels = labels.view(-1, 1)
        
                # Forward pass
                rgb_outputs = i3d_rgb(rgb_data)
                flow_outputs = i3d_flow(flow_data)
        
                # Combine the outputs if necessary. Here, we'll average, but other strategies can be used.
                combined_outputs = (rgb_outputs + flow_outputs) / 2.0
        
                # Calculate loss
                loss = criterion(combined_outputs, labels)
        
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # Save models
        torch.save(i3d_rgb.state_dict(), 'i3d_rgb_model.pth')
        torch.save(i3d_flow.state_dict(), 'i3d_flow_model.pth')

    else:

        model,train_dataloader,test_dataloader=trainer(20,5,240,30,transform_train,transform_test,mode)
        model=torch.load('model_path_fine_tuned_1.pth').to(mps_device)
        criterion = nn.CrossEntropyLoss()

        max_lr = 0.0001
        epochs = 1
        lrs=[]
        train_losses=[]
        test_losses=[]
        #optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=1e-4) 
        optimizer = torch.optim.Adam(model.parameters(), max_lr,weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dataloader))
        # Training loop
        for epoch in range(epochs):  # loop over the dataset multiple times
            model.train()  # Set model to training mode
            train_preds, train_labels,train_titles,train_categories,train_prompts = [], [],[],[],[]
            running_loss = 0.0
            for i, data in enumerate(tqdm(train_dataloader)):
                inputs, labels,title, category, prompt= data
                inputs, labels = inputs.float().to(mps_device), labels.to(mps_device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.2)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                train_titles.extend(title)
                train_categories.extend(category)
                train_prompts.extend(prompt)

            train_accuracy = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='macro')
            train_precision = precision_score(train_labels, train_preds, average='macro')
            train_recall = recall_score(train_labels, train_preds, average='macro')

            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader)}, Train Accuracy: {train_accuracy}, Train F1: {train_f1}, Precision: {train_precision}, Recall: {train_recall}")

            with open('train_results2.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                for prompt,category,title, actual, predicted in zip(train_prompts,train_categories,train_titles, train_labels, train_preds):
                    writer.writerow([prompt,category,title, actual, predicted])

            model.eval()  # Set model to evaluation mode
            correct = 0
            total = 0
            test_preds, test_labels,test_titles,test_categories,test_prompts = [], [],[],[],[]
            val_loss = 0.0
            with torch.no_grad():
                for data in test_dataloader:
                    inputs, labels, title,category,prompt= data
                    inputs, labels = inputs.float().to(mps_device), labels.to(mps_device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    test_preds.extend(predicted.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
                    test_titles.extend(title)
                    test_categories.extend(category)
                    test_prompts.extend(prompt)

            test_accuracy = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average='macro')
            test_precision = precision_score(test_labels, test_preds, average='macro')
            test_recall = recall_score(test_labels, test_preds, average='macro')

            print(f"Epoch {epoch+1}, Test Loss: {val_loss/len(test_dataloader)}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall}")

            with open('test_results2.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                for prompt,category,title, actual, predicted in zip(test_prompts,test_categories,test_titles, test_labels, test_preds):
                    writer.writerow([prompt,category,title, actual, predicted])

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

    