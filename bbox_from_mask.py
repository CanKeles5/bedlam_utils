'''
Create bbox from segmentation mask
'''

import cv2
import numpy as np


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
    -
    
    Args:
        -
    
    Returns:
        -
    '''
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for contour in contours:
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x+w, y+h))  # Format: (x_min, y_min, x_max, y_max)
    
    return bounding_boxes

# Example usage
# Assuming 'mask' is a binary mask image where white pixels represent the person

#mask = cv2.imread('person_mask.png', cv2.IMREAD_GRAYSCALE)  # Load the mask
b_mask = cv2.imread('seq_000000_0000_00_body.png', cv2.IMREAD_GRAYSCALE)
c_mask = cv2.imread('seq_000000_0000_00_clothing.png', cv2.IMREAD_GRAYSCALE)

print(f"b_mask.shape: {b_mask.shape}, c_mask.shape: {c_mask.shape}")

mask = join_masks(b_mask, c_mask)

print(f"mask.shape: {mask.shape}")

bounding_boxes = get_bounding_box(mask)

print(f"bounding_boxes: {bounding_boxes}")

### Visualization ###

def draw_bounding_boxes(image, bounding_boxes):
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw rectangle
        
    return image


image_with_boxes = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

image_with_boxes = draw_bounding_boxes(image_with_boxes, bounding_boxes)

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################

