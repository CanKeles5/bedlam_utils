'''
Iterate over depth & rgb images.
Convert the depth values from exr to png.
Save the depth & imgs in a single new folder
Also create the necessary txt file
'''


import pyexr
import numpy as np
import cv2
import os
import glob
import shutil


def exr_to_nyu_depth_v2(exr_path, png_output_path, mask_png):
    # Read the EXR image
    exr_data = pyexr.open(exr_path)

    # Extract the channel representing depth (you may need to adjust this based on your specific EXR file)
    #depth_channel = exr_data.get("Z")
    depth_channel = exr_data.get()[..., 0]

    # Convert the depth channel to a numpy array
    depth_np = np.array(depth_channel)
    
    # Map the depth values to the 8-bit range (0 to 255)
    #depth_uint8 = np.clip((depth_np / depth_np.max()) * 255, 0, 255).astype(np.uint8)
    depth_uint16 = (depth_np).astype(np.uint16)
    
    #mask_png = 1 - mask_png
    #depth_uint16 *= mask_png
    
    # Define the new width and height
    new_width = 512
    new_height = 384
    
    # Resize the depth image
    resized_depth_image = cv2.resize(depth_uint16, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Save the depth map as PNG (you may need to adjust the image format based on your use case)
    cv2.imwrite(png_output_path, resized_depth_image)


if __name__ == '__main__':
    
    #instead of seperate exr & png roots, we can have the root pth and we can create the exr & png paths

    folder_pths = ["/home/cankeles/be_imagedata_download/20221011_1_250_batch01hand_closeup_suburb_a", "/home/cankeles/be_imagedata_download/20221010_3-10_500_batch01hand_zoom_suburb_d", "/home/cankeles/be_imagedata_download/20221010_3_1000_batch01hand"]
    data_root = r"/home/cankeles/be_imagedata_download/20221010_3_1000_batch01hand"

    exr_root = os.path.join(data_root, "depth")
    png_root = os.path.join(data_root, "png")
    mask_root = os.path.join(data_root, "masks")
    
    new_data_root = r"/home/cankeles/be_imagedata_download/data_fixed"
    
    #Originally, BEDLAM is 30 fps
    des_fps = 6
    
    for exr_seq in sorted(glob.glob(os.path.join(exr_root, "*"))):
        cfps = 0
        
        #Create the folder for the new sequence
        new_seq_dir = os.path.join(new_data_root, f"{os.path.normpath(exr_seq).split(os.sep)[-3]}_{os.path.split(exr_seq)[-1]}")
        os.mkdir(new_seq_dir)
        
        for exr_pth in sorted(glob.glob(os.path.join(exr_seq, "*.exr"))):
            
            #print(f"exr_path: {os.path.split(exr_pth)[-1][:-9] + 'env.png'}")
            
            if cfps % des_fps==0:
                id = exr_pth.split('_')[-2]
                depth_pth = os.path.join(new_seq_dir, f"sync_depth_{id}.png")
                
                #mask_pth = os.path.join(mask_root, os.path.split(exr_seq)[-1], f"{os.path.split(exr_pth)[-1][:-9]}env.png")

                #mask_png = cv2.imread(mask_pth)
                
                #All 3 channels of the mask are equal
                exr_to_nyu_depth_v2(exr_pth, depth_pth, []) #mask_png[:, :, 0])
                
            cfps +=1
    
    #print(f"Done with exr files, proceeding to pngs.")
    for png_seq in sorted(glob.glob(os.path.join(png_root, "*"))):
        cfps = 0
        
        #Create the folder for the new sequence
        new_seq_dir = os.path.join(new_data_root, f"{os.path.normpath(png_seq).split(os.sep)[-3]}_{os.path.split(png_seq)[-1]}")
        
        for png_pth in sorted(glob.glob(os.path.join(png_seq, "*.png"))):
            
            if cfps % des_fps==0:
                id = png_pth.split('_')[-1]
                
                new_png_pth = os.path.join(new_seq_dir, f"rgb_{id}")
                
                rgb_image = cv2.imread(png_pth)

                #Set desired image size
                new_width = 512
                new_height = 384
                
                # Resize the RGB image
                resized_rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(new_png_pth, resized_rgb_image)
                
                #shutil.copy(png_pth, new_png_pth)
        
            cfps += 1

