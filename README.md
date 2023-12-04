# AoTModelScopeT2V
This repository is includes the files for the project to answer the question can ModelScopeT2V understand the Arrow of Time?

In this project we analyze if ModelScopeT2V is capable of generating videos that follow the Arrow of Time.
Please see the paper here: 


We created a classifier that is capable of classifying videos shown forward and backward. 

It was trained on UCF-101 dataset which can be accessed here: https://www.crcv.ucf.edu/data/UCF101.php

Results of the project are available in the results folder.

To use the file for training Final_train.py , ensure you have videos prepared in the following format:

- Root Folder
  - Category Folders
    - Prompt Length
      - Video Folder
        - Video Images

You also need to extract 10 frames from your video. If you want to use more than 10 frames or less, you need to adjust the dataset and model definition in Final_train.py.

Pre-trained Models: https://drive.google.com/drive/u/0/folders/1t_Ow9-R-RsPJLJXLs--tVYPfqJxiPPod

ModelScopeT2V: https://huggingface.co/damo-vilab/text-to-video-ms-1.7b

To generate GradCAM images of your result, please use Final_Analysis.py

The MISC includes a sandbox file which has experimental environment used to try different architectures.
