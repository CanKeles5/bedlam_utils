
'''
Create txt file for train & test.

Format: "FOLDER_NAME/rgb_0000.png FOLDER_NAME/sync_depth_0000.png FOCAL_LENGTH"

todo:
- Add processing of env masks and add them to the txt files.
'''

import glob
import os
import re
import pandas as pd
import math


def remove_leading_zeros(input_str):
    
    if input_str == '0' * len(input_str):
        return 0
    
    return int(input_str)


def compute_focal_length(original_image_width, resized_image_width, hfov_deg):
    hfov_rad = math.radians(hfov_deg)
    focal_length_resized = resized_image_width / (2 * math.tan(hfov_rad / 2))
    
    return focal_length_resized



if __name__ == '__main__':
    
    data_root = r"/home/cankeles/be_imagedata_download/eval_dataset"
    csv_root = r"/home/cankeles/be_imagedata_download/bedlam_fc/20221019_3_250_highbmihand/ground_truth/camera"
    txt_path = r"new_test_paths.txt"
    
    target_substring = r"20221019_3_250_highbmihand"

    with open(txt_path, "a") as file:
        for folder in filter(lambda filename: target_substring in filename, sorted(glob.glob(os.path.join(data_root, "*")))):
            
            contents = glob.glob(os.path.join(folder, "*"))
            
            str = os.path.split(contents[0])[-1].split("_")
            
            csv_folder = f"{os.path.split(folder)[-1].split('_')[-2]}_{os.path.split(folder)[-1].split('_')[-1]}_camera.csv"
            csv_pth = os.path.join(csv_root, csv_folder)
            
            data = pd.read_csv(csv_pth)
            #print(f"data[0]: {data.iloc[0]}")
            
            pattern = r"(\d+)\.png"
            
            for pth in contents:
                
                if os.path.split(pth)[-1][:3]=='rgb':
                    match = re.search(pattern, pth)
                    if match:
                        number_before_png = match.group(1)
                        
                        png_pth = os.path.join(os.path.split(folder)[-1], f"rgb_{number_before_png}.png")
                        depth_pth = os.path.join(os.path.split(folder)[-1], f"sync_depth_{number_before_png}.png")
                        mask_pth = os.path.join(os.path.split(folder)[-1], f"mask_{number_before_png}.png")
                        
                        hfov = data.iloc[remove_leading_zeros(number_before_png)]['hfov']
                        width = 1280
                        
                        new_line = "\n"
                        
                        focal_length = compute_focal_length(1280, 512, hfov)
                        
                        file.write(f"{png_pth} {depth_pth} {mask_pth} {focal_length}{new_line}")

