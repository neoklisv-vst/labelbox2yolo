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


dataset_name = 'labelbox'


# Define your directories
source_directory = './' + dataset_name + '/train'
test_directory = './' + dataset_name + '/test'
val_directory = './' + dataset_name + '/val'

# Define source directories
source_images_dir = dataset_name + '/images'
source_labels_dir = dataset_name + '/labels'

# Define target directories
train_images_dir = dataset_name + '/train/images'
train_labels_dir = dataset_name + '/train/labels'
test_images_dir = dataset_name + '/test/images'
test_labels_dir = dataset_name + '/test/labels'
val_images_dir = dataset_name + '/val/images'
val_labels_dir = dataset_name + '/val/labels'

# List of all target directories to create
directories = [
    train_images_dir, train_labels_dir,
    test_images_dir, test_labels_dir,
    val_images_dir, val_labels_dir
]

# Create all target directories if they don't exist
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Function to move files from source to destination
def move_files2(source_dir, target_dir):
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        shutil.move(source_file, target_file)

# Move image files
move_files2(source_images_dir, train_images_dir)

# Move label files
move_files2(source_labels_dir, train_labels_dir)

# Function to delete a directory if it is empty
def delete_directory(directory):
    if os.path.exists(directory) and len(os.listdir(directory)) == 0:
        os.rmdir(directory)

# Delete original directories if they are empty
delete_directory(source_images_dir)
delete_directory(source_labels_dir)


print("Files moved and directories created successfully.")

# Run the function
main(source_directory, test_directory, val_directory)
