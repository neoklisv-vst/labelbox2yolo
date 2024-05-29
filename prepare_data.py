import os
import shutil
from random import sample

def move_files(source_dir, target_dir, files):
    for file in files:
        # Move the image file
        shutil.move(os.path.join(source_dir,'images', file), os.path.join(target_dir,'images', file))
        # Move the corresponding label file (replace .jpg with .txt)
        shutil.move(os.path.join(source_dir,'labels', file.replace('.png', '.txt')), os.path.join(target_dir,'labels', file.replace('.png', '.txt')))
       

def main(source_dir, test_dir, val_dir):
    # List all .jpg files in the source directory
    all_images = [f for f in os.listdir(os.path.join(source_dir,'images')) if f.endswith('.png')]
    # print(all_images)    
    
    # Calculate 10%(15%) of the total number of images
    ten_percent = int(len(all_images) * 0.15)
    print(ten_percent)
    
    # Randomly select 10% of the images for the test set
    test_files = sample(all_images, ten_percent)
    
    # Remove the selected test files from the list of all images
    remaining_images = [img for img in all_images if img not in test_files]
    
    # Randomly select another 10% of the remaining images for the validation set
    val_files = sample(remaining_images, ten_percent)
    
    # Move the selected files to their respective directories
    move_files(source_dir, test_dir, test_files)
    move_files(source_dir, val_dir, val_files)

# Define your directories
source_directory = './demoroom_5500/train'
test_directory = './demoroom_5500/test'
val_directory = './demoroom_5500/val'

# Run the function
main(source_directory, test_directory, val_directory)
