import os
import cv2

# Directory containing images
input_dir = "midlothian-1379/images"
output_dir = "midlothian-1379/grayscale_images_cv"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image formats
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_extensions):
        img_path = os.path.join(input_dir, filename)
        try:
            # Read image using OpenCV
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {filename}: Unable to read image")
                continue
            
            # Convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            # Save the grayscale image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, gray_img)
            print(f"Converted: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Conversion complete!")
