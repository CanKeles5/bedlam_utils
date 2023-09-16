import os
import shutil

# Specify the paths of the source folders and the destination folder
source_folders = ['/home/cankeles/be_imagedata_download/new_20221011_1_250_batch01hand_closeup_suburb_a', '/home/cankeles/be_imagedata_download/new_20221010_3-10_500_batch01hand_zoom_suburb_d', '/home/cankeles/be_imagedata_download/new_20221010_3_1000_batch01hand']
destination_folder = '/home/cankeles/be_imagedata_download/joined_data'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through each source folder
for source_folder in source_folders:
    # Iterate through each item (subfolder) in the source folder
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Create a destination path based on the item name
            destination_path = os.path.join(destination_folder, item)
            
            # Copy the subfolder to the destination folder
            shutil.copytree(item_path, destination_path)

print("Subfolders copied successfully!")
