import cv2
import os
import numpy as np

# Define the image augmentation methods


def augment_image(img):
    augmented_images = []

    # Flip the image horizontally
    augmented_images.append(cv2.flip(img, 1))

    # Rotate the image by 45 degrees
    M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), 45, 1)
    augmented_images.append(cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0])))

    # Apply Gaussian blur
    augmented_images.append(cv2.GaussianBlur(img, (5, 5), 0))

    # Add random noise
    # Ensure dtype is the same as img
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    augmented_images.append(cv2.add(img, noise))

    # Increase brightness
    bright_img = np.clip(img * 1.2, 0, 255).astype(np.uint8)
    augmented_images.append(bright_img)

    # Decrease brightness
    dark_img = np.clip(img * 0.8, 0, 255).astype(np.uint8)
    augmented_images.append(dark_img)

    # Add spotlights
    mask_center = np.zeros(img.shape[:2], dtype=np.uint8)
    mask_edges = np.zeros(img.shape[:2], dtype=np.uint8)

    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    cv2.circle(mask_center, (center_x, center_y), 100, 255, -1)
    cv2.circle(mask_edges, (center_x, center_y), 100, 255, -1)

    edge_radius = min(center_x, center_y)
    cv2.circle(mask_edges, (center_x, center_y), edge_radius, 0, -1)

    spotlight_center = cv2.bitwise_and(img, img, mask=mask_center)
    spotlight_edges = cv2.bitwise_and(img, img, mask=mask_edges)

    augmented_images.extend([spotlight_center, spotlight_edges])

    return augmented_images


def augment():
    current_directory = os.getcwd()
    src_dir = os.path.join(current_directory, "dataset")

    dest_dir = current_directory
    raw_dir = os.path.join(current_directory, "raw")

    # Create raw directory if it doesn't exist
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    # Process each folder (id) in the source directory
    for folder in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder)

        if os.path.isdir(folder_path):
            # Create corresponding id folder in the raw directory
            raw_folder_path = os.path.join(raw_dir, folder)
            if not os.path.exists(raw_folder_path):
                os.makedirs(raw_folder_path)

            # Process each image in the id folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if os.path.isfile(file_path):
                    # Read the image
                    img = cv2.imread(file_path)

                    # Augment the image
                    augmented_images = augment_image(img)

                    # Save the augmented images to the raw folder
                    for i, augmented_img in enumerate(augmented_images):
                        dest_file_path = os.path.join(
                            raw_folder_path, f"{file_name.split('.')[0]}_augmented_{i+1}.jpg")
                        cv2.imwrite(dest_file_path, augmented_img)

    # Create or update a file to keep track of augmented ids
    augmented_ids_file = os.path.join(dest_dir, "augmented_ids.txt")
    with open(augmented_ids_file, "a") as f:
        f.write("\n".join(os.listdir(raw_dir)))

    print("Image augmentation completed!")


if __name__ == "__main__":
    augment()
