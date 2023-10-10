'''
Iterate over depth, rgb, masks.
Create a dataset where each sample contains a single person instance.
'''


import pyexr
import numpy as np
import cv2
import os
import glob
import shutil

import bbox_from_mask

#Change the image size!
NEW_WIDTH, NEW_HEIGHT = (512, 384)


if __name__ == '__main__':
    
    #instead of seperate exr & png roots, we can have the root pth and we can create the exr & png paths
    
    folder_pths = ['/home/cankeles/be_imagedata_download/data/20221019_3_250_highbmihand']
    #data_root = r"/home/cankeles/be_imagedata_download/20221010_3-10_500_batch01hand_zoom_suburb_d"
    new_data_root = r"/home/cankeles/be_imagedata_download/eval_dataset"
    
    for folder in folder_pths:
        #exr_root = os.path.join(folder, "depth")
        png_root = os.path.join(folder, "png")
        mask_root = os.path.join(folder, "masks")
        
        #Originally, BEDLAM is 30 fps
        des_fps = 6
        
        for (png_seq, mask_seq) in zip(sorted(glob.glob(os.path.join(png_root, "*"))), sorted(glob.glob(os.path.join(mask_root, "*")))):
            
            #ASSERT IF FOLDER NAMES ARE THE SAME!!!
            assert os.path.split(png_seq)[-1] == os.path.split(mask_seq)[-1], "Folder names of png and mask sequences dont match."
            
            cfps = 0
            
            #Create the folder for the new sequence
            new_seq_dir = os.path.join(new_data_root, f"{os.path.normpath(png_seq).split(os.sep)[-3]}_{os.path.split(png_seq)[-1]}")
            #os.mkdir(new_seq_dir)

            for png_pth in sorted(glob.glob(os.path.join(png_seq, "*.png"))):
                
                if cfps % des_fps==0:
                    id = png_pth.split('_')[-1][:-4]

                    new_png_pth = os.path.join(new_seq_dir, f"rgb_{id}.png")
                    
                    rgb_image = cv2.imread(png_pth)
                    
                    # Resize the RGB image
                    resized_rgb_image = cv2.resize(rgb_image, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    #cv2.imwrite(new_png_pth, resized_rgb_image)
                    
                    
                    ### Get the required masks of each instance ###
                    
                    mask_body_paths = sorted(glob.glob(os.path.join(mask_seq, f'{os.path.splitext(os.path.basename(image_path))[0]}_*_body.png'))[0])
                    mask_clothing_paths = sorted(glob.glob(os.path.join(mask_seq, f'{os.path.splitext(os.path.basename(image_path))[0]}_*_clothing.png'))[0])
                    
                    print(f"mask_body_path: {mask_body_paths}")
                    print(f"mask_clothing_paths: {mask_clothing_paths}")
                    
                    for (b_pth, c_pth) in zip(mask_body_paths, mask_clothing_paths):
                        b_mask = cv2.imread(b_pth, cv2.IMREAD_GRAYSCALE)
                        c_mask = cv2.imread(c_pth, cv2.IMREAD_GRAYSCALE)
                        
                        mask = bbox_from_mask.join_masks(b_mask, c_mask)
                        bbox = bbox_from_mask.get_bounding_box(mask)
                        
                        #Get only the bbox region (ie only the person instance) from the RGB image & mask
                        img_crop = None
                        mask_crop = None
                        
                        #Resize the image & masks if necessary
                        resized_img_crop = cv2.resize(img_crop, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_LINEAR)
                        resized_mask_crop = cv2.resize(mask_crop, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_LINEAR)
                        
                        cv2.imwrite(new_png_pth, resized_img_crop)
                        cv2.imwrite(new_mask_pth, resized_mask_crop)
                    
                    ###############################################
            
                cfps += 1
        
        """
        for mask_seq in sorted(glob.glob(os.path.join(mask_root, "*"))):
            cfps = 0

            new_seq_dir = os.path.join(new_data_root, f"{os.path.normpath(mask_seq).split(os.sep)[-3]}_{os.path.split(mask_seq)[-1]}")
            #os.mkdir(new_seq_dir)
            for mask_pth in sorted(glob.glob(os.path.join(mask_seq, "*_env.png"))):
                
                if cfps % des_fps==0:
                    id = mask_pth.split('_')[-2]
                    
                    new_png_pth = os.path.join(new_seq_dir, f"mask_{id}.png")
                    
                    mask_image = cv2.imread(mask_pth)
                    
                    inverted_image = np.where(mask_image == 0, 65535, 0).astype(np.uint16)
                    
                    # Resize the RGB image
                    resized_mask = cv2.resize(inverted_image, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(new_png_pth, resized_mask)
        """

