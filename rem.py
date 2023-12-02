import os
from PIL import Image
from tqdm import tqdm 

streamorder_directory = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/rgb2so/so_data_64/'
rgb_directory = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/rgb2so/rgb_data_64/'
    
def remove_mismatched_images(directory, associated_directory, file_prefix):
    # List all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files):
            if file.startswith(file_prefix) and file.endswith(".png"):
                file_path = os.path.join(root, file)
                img = Image.open(file_path)
                if img.size != (64, 64):
                    # Remove the image from the current directory
                    os.remove(file_path)

                    # Get the image number from the filename
                    image_number = file.split('_')[-1].split('.')[0]

                    # Remove corresponding images in the associated directory
                    for assoc_root, assoc_dirs, assoc_files in os.walk(associated_directory):
                        for assoc_file in assoc_files:
                            if assoc_file.endswith(f"{image_number}.png"):
                                assoc_file_path = os.path.join(assoc_root, assoc_file)
                                os.remove(assoc_file_path)

# Process the streamorder directory and remove any associated images in the rgb directory
remove_mismatched_images(streamorder_directory, rgb_directory, "tile_streamorder_")

print("First Done")

# # Process the rgb directory and remove any associated images in the streamorder directory
remove_mismatched_images(rgb_directory, streamorder_directory, "tile_")

print("Image cleanup is done!")