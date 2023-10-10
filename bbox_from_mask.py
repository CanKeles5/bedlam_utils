'''
Create bbox from segmentation mask
'''

import cv2
import numpy as np
import os
import glob


def join_masks(b_mask, c_mask):
    '''
    Joins the body and clothing masks for instances in the BEDLAM dataset.

    Args:
        b_mask (numpy.array): Body mask of instance.
        c_mask (numpy.array): Clothing mask of instance.
    
    Returns:
        mask: Full body mask of instance.

    '''
    
    mask = b_mask + c_mask
    #mask = None #Apply a threshold on the vals. There might not be a need for this as the masks shold not overlap at all. Check.
    
    return mask
    

def get_bounding_box(mask):
    '''
    Creates a bbox from the provided mask for a single instance.
    
    Args:
        mask (numpy.array): Full body mask of a single instance.
    
    Returns:
        bounding_boxes (List): A list containing a single bbox for the instance.
    '''
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for contour in contours:
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        #w = x_max - x_min
        #h = y_max - y_min
        x_min = x
        x_max = x + w
        y_min = y
        y_max = y + h
        
        size = int(max(w*1.1, h*1.1))
        
        
        # Calculate the coordinates for the top-left corner of the square bounding box
        new_x_min = x_min + (w - size) // 2
        new_y_min = y_min + (h - size) // 2
        new_x_max = new_x_min + size
        new_y_max = new_y_min + size
        
        increase_w = int(size * 0.1)
        increase_h = int(size * 0.1)

        # Calculate the new coordinates for the enlarged bounding box
        new_x_min = max(0, new_x_min - increase_w // 2)
        new_y_min = max(0, new_y_min - increase_h // 2)
        new_x_max = min(w, new_x_max + increase_w // 2)
        new_y_max = min(h, new_y_max + increase_h // 2)

        bounding_boxes.append((new_x_min, new_y_min, new_x_max, new_y_max, size))
        #bounding_boxes.append((x, y, x+w, y+h))  # Format: (x_min, y_min, x_max, y_max)
    
    return bounding_boxes


def draw_bounding_boxes(image, bounding_boxes):

    #box = bounding_boxes[0]

    print(f"bounding_boxes: {bounding_boxes}")

    #for box in bounding_boxes:
    
    x_min, y_min, x_max, y_max, size = bounding_boxes
    
    w = x_max - x_min
    h = y_max - y_min
    
    '''
    # Calculate the coordinates for the top-left corner of the square bounding box
    new_x_min = x_min + (w - size) // 2
    new_y_min = y_min + (h - size) // 2
    new_x_max = new_x_min + size
    new_y_max = new_y_min + size
    '''
    
    #cv2.rectangle(image, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 255, 0), 2)
    
    cv2.rectangle(image, (x_min, y_min), (x_min+size, y_min+size), (0, 255, 0), 2)  # Draw rectangle
    
    #image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    #Usage for visualizing a single instance & bbox
    #image_with_boxes = draw_bounding_boxes(image_with_boxes, bounding_boxes)
    
    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.imshow('Cropped image', image[y_min:y_min+size, x_min:x_min+size])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return image


"""
# Example usage
# Assuming 'mask' is a binary mask image where white pixels represent the person

#mask = cv2.imread('person_mask.png', cv2.IMREAD_GRAYSCALE)  # Load the mask
b_mask = cv2.imread('seq_000000_0000_00_body.png', cv2.IMREAD_GRAYSCALE)
c_mask = cv2.imread('seq_000000_0000_00_clothing.png', cv2.IMREAD_GRAYSCALE)

mask = join_masks(b_mask, c_mask)

bounding_boxes = get_bounding_box(mask)
"""

if __name__ == '__main__':
    
    root = r"C:\Users\Can\Desktop\bedlam_utils\toy_data"
    png_root = os.path.join(root, "png")
    mask_root = os.path.join(root, "masks")
    
    new_data_root = r"C:\Users\Can\Desktop\bedlam_utils\toy_data\crop"
    
    for png_seq in sorted(glob.glob(os.path.join(png_root, "*"))):
        cfps = 0
        
        #Create the folder for the new sequence
        new_seq_dir = os.path.join(new_data_root, f"{os.path.normpath(png_seq).split(os.sep)[-3]}_{os.path.split(png_seq)[-1]}")
        print(f"new_seq_dir: {new_seq_dir}")
        
        #os.mkdir(new_seq_dir) #might not be necessary

        for png_pth in sorted(glob.glob(os.path.join(png_seq, "*.png"))):
            
            des_fps = 1
            
            if cfps % des_fps==0:
                id = png_pth.split('_')[-1][:-4]
                
                #new_png_pth = os.path.join(new_seq_dir, f"rgb_{id}.png")
                
                rgb_image = cv2.imread(png_pth)
                
                print(f"png_pth: {png_pth}")
                
                #Set desired image size
                #new_width = 512
                #new_height = 384
                
                # Resize the RGB image
                #resized_rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                #cv2.imwrite(new_png_pth, resized_rgb_image)
                
                #shutil.copy(png_pth, new_png_pth)
                
                #png_pth_root = png_pth[:-4]
                
                #print(f"png_pth: {png_pth}")
                #print(f"png_pth: {os.path.split(png_pth)}")
                
                ###
                prefix, num, suffix = os.path.split(png_pth)[-1].split('_')                
                
                # Pad the numeric part with leading zeros if needed
                num = num.zfill(6)
                
                
                # Generate the new strings for pngs
                for i in range(50):  # Generating 10 strings as an example
                    new_string_body = f"{prefix}_{num}_{suffix[:-4]}_{str(i).zfill(2)}_body.png"
                    new_string_clothing = f"{prefix}_{num}_{suffix[:-4]}_{str(i).zfill(2)}_clothing.png"
                    
                    dirname = png_seq.replace("png", "masks")
                    
                    if os.path.exists(os.path.join(dirname, new_string_body)):
                        body_mask = cv2.imread(os.path.join(dirname, new_string_body), cv2.IMREAD_GRAYSCALE)
                        clothing_mask = cv2.imread(os.path.join(dirname, new_string_clothing), cv2.IMREAD_GRAYSCALE)
                        
                        joined_mask = join_masks(body_mask, clothing_mask)
                        bbox = get_bounding_box(joined_mask)[0]
                        
                        cropped_instance = rgb_image[bbox[1]:bbox[1]+bbox[4], bbox[0]:bbox[0]+bbox[4]]
                        
                        print(f"cropped_instance shape: {cropped_instance.shape}")
                        
                        crop_pth = f"{png_pth[:-4]}_{str(i).zfill(2)}.png"
                        crop_pth = crop_pth.replace("\\png\\", "\\crop\\")
                        print(f"i: {i}, crop_pth: {crop_pth}")
                        
                        cv2.imwrite(crop_pth, cropped_instance)

                        #draw_bounding_boxes(rgb_image, bbox)
                        
                        print(f"bbox: {bbox}")
                    else:
                        break
                ###
                
                
            cfps += 1

