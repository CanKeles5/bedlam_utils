'''
Take the avg of the lower half of the human body
'''

import cv2
import numpy as np

def calculate_average_depth_lower_half(depth_map, mask):
    region_of_interest = np.where(mask)
    depths_in_roi = depth_map[region_of_interest]
    
    print(f"region_of_interest.shape: {region_of_interest[0].shape}")
    
    height = np.max(region_of_interest[0]) - np.min(region_of_interest[0])
    
    print(f"height: {height}")
    
    midpoint_y = (np.min(region_of_interest[0]) + np.max(region_of_interest[0])) // 2
        
    lower_half_indices = region_of_interest[0] >= midpoint_y
    depths_lower_half = depths_in_roi[lower_half_indices]
    average_depth = np.mean(depths_lower_half)
    return average_depth, height


def calculate_average_depth(original_depth_map, mask):
    # Extract depths within the region of interest
    region_of_interest = np.where(mask)
    depths_in_roi = original_depth_map[region_of_interest]
    
    # Calculate the average depth within the region
    average_depth = np.mean(depths_in_roi)
    
    return average_depth


def invert_depth_in_mask_lower_half(depth_map, mask):
    # Invert depth values only within the masked region
    inverted_depth = depth_map.copy()
    inverted_depth[mask] = np.max(depth_map) - inverted_depth[mask]
    return inverted_depth

def replace_human_depth(scene_depth, human_depth):
    _, human_mask = cv2.threshold(human_depth, 1, 255, cv2.THRESH_BINARY)
    background_mask = cv2.bitwise_not(human_mask)

    # Calculate the average depth of the lower half of humans
    average_human_depth, height = calculate_average_depth_lower_half(scene_depth, human_mask)

    # Invert depth values within the region where replacement occurs
    #inverted_human_depth = invert_depth_in_mask_lower_half(human_depth, human_mask)
    
    avg_hdepth = calculate_average_depth(human_depth, human_mask)
    
    human_depth = human_depth.astype(np.float32)

    # Calculate the displacement matrix within the region
    displacement_matrix = human_depth - avg_hdepth

    # Apply the mask to the displacement matrix
    displacement_matrix[human_mask == False] = 0

    human_depth = average_human_depth + displacement_matrix * (average_human_depth / avg_hdepth)

    # Replace the corresponding values in the scene depth map with the inverted depth
    replaced_depth = np.where(human_mask != 0, human_depth, scene_depth)

    return replaced_depth


# Example usage
scene_depth_map = cv2.imread("image_00995.jpgorig_bedlam_depth.png", cv2.IMREAD_UNCHANGED) #, cv2.IMREAD_GRAYSCALE).astype(np.uint16) # / 255.0
human_depth_map = cv2.imread("depth_map.png", cv2.IMREAD_UNCHANGED) #cv2.IMREAD_GRAYSCALE).astype(np.uint8) #/ 255.0

#print(scene_depth_map)
#print(human_depth_map)

print(f"scene depth min: {scene_depth_map.min()}, max: {scene_depth_map.max()}, type: {scene_depth_map.dtype}, size: {scene_depth_map.shape}")
print(f"human depth min: {human_depth_map.min()}, max: {human_depth_map.max()}, type: {human_depth_map.dtype}, size: {human_depth_map.shape}")


# Replace human depths in the scene depth map
result_depth_map = replace_human_depth(scene_depth_map, human_depth_map).astype(np.uint16)
cv2.imwrite("result_depth_map.png", result_depth_map) #.astype(np.float32))

'''
# Display the original and replaced depth maps
cv2.imshow("Scene Depth Map", scene_depth_map)
cv2.imshow("Human Depth Map", human_depth_map)
cv2.imshow("Result Depth Map", result_depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

