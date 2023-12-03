#Import Statements
import cv2
import os
import numpy as np
'''
Example of Extraction of Frames from a Video
'''
def extract_frames(video_path, output_folder, num_frames=10):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Open the video
    cap = cv2.VideoCapture(video_path)
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Select equally spaced frames
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=np.int)
    # Iterate through frames
    for i, frame_idx in enumerate(frame_indices):
        # Set the frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # Read the frame
        ret, frame = cap.read()
        # Save the frame
        if ret:
            cv2.imwrite(os.path.join(output_folder, f'{i}.png'), frame)
        else:
            print(f"Failed to extract frame at index {frame_idx} from {video_path}")
    # Release the video capture object
    cap.release()
    
# Call the Extraction Frames in this function
def process_videos(root_dir):
    # Iterate through all categories
    for category in os.listdir(root_dir):
        # Iterate through all subfolders (backward, forward)
        category_path = os.path.join(root_dir, category)
        # Iterate through all prompts
        if os.path.isdir(category_path):
            # Iterate through all videos in forward and backward
            for subfolder in ['0', '1']:
                # Iterate through all videos in forward and backward
                subfolder_path = os.path.join(category_path, subfolder)
                if os.path.isdir(subfolder_path):
                    # Iterate through all types of prompts
                    for prompt in os.listdir(subfolder_path):
                        prompt_path = os.path.join(subfolder_path, prompt)
                        if os.path.isdir(prompt_path):
                            # Iterate through all videos in prompt folder
                            for video_file in os.listdir(prompt_path):
                                # Extract frames from video
                                if video_file.endswith('.mp4'):
                                    # Get video path, name and output folder
                                    video_path = os.path.join(prompt_path, video_file)
                                    video_name = os.path.splitext(video_file)[0]
                                    output_folder = os.path.join(prompt_path, video_name)
                                    if not os.path.exists(output_folder):
                                        print(f"Processing {video_path}")
                                        extract_frames(video_path, output_folder)
                                    else:
                                        print(f"Skipping {video_path}, output folder already exists.")

# Usage
root_dir = './Videos1_f'
process_videos(root_dir)
