import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List

def save_cifar10_batches_as_images(batch_files: List[Path], output_dir: Path) -> None:
    """
    Extracts images from CIFAR-10 batch files and saves them as PNG images with their original filenames.

    Args:
        batch_files (list): List of paths to CIFAR-10 batch files (pickled).
        output_dir (str): Path to the output directory where images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for batch_idx, batch_file in enumerate(batch_files):
        # Load the batch file
        batch = unpickle(batch_file)
        data = batch[b'data']  # Extract image data
        labels = batch[b'labels']  # Extract labels
        filenames = batch[b'filenames']  # Extract filenames (bytes)

        if "test" in batch_file:
            batch_folder = os.path.join(output_dir, 'test_batch')
        else:
            batch_folder = os.path.join(output_dir, f'batch_{batch_idx + 1}')
        os.makedirs(batch_folder, exist_ok=True)

        # Process each image in the batch
        for i, image_data in enumerate(data):
            # Convert the flat data into 3D (32x32x3)
            image_r = image_data[:1024].reshape(32, 32)
            image_g = image_data[1024:2048].reshape(32, 32)
            image_b = image_data[2048:].reshape(32, 32)
            image = np.stack([image_r, image_g, image_b], axis=2)

            # Convert to PIL Image and save as PNG
            img = Image.fromarray(image)
            original_filename = filenames[i].decode('utf-8')  # Decode bytes to string
            img_filepath = os.path.join(batch_folder, original_filename)
            img.save(img_filepath)

        print(f"Batch {batch_idx + 1} processed and saved to {batch_folder}")

def unpickle(file_name: Path) -> dict:
    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_images():
    batch_files = [
            Path('/home/yonatan.r/Desktop/CS/Deep Learning/CV_PROJECT/cifar-10-batches-py/data_batch_1'),
            Path("/home/yonatan.r/Desktop/CS/Deep Learning/CV_PROJECT/cifar-10-batches-py/data_batch_2"),
            Path("/home/yonatan.r/Desktop/CS/Deep Learning/CV_PROJECT/cifar-10-batches-py/data_batch_3"),
            Path("/home/yonatan.r/Desktop/CS/Deep Learning/CV_PROJECT/cifar-10-batches-py/data_batch_4"),
            Path("/home/yonatan.r/Desktop/CS/Deep Learning/CV_PROJECT/cifar-10-batches-py/data_batch_5"),
            Path("/home/yonatan.r/Desktop/CS/Deep Learning/CV_PROJECT/cifar-10-batches-py/test_batch")
        ]
    output_dir = Path('/home/yonatan.r/Desktop/CS/Deep Learning/CV_PROJECT/cifar10_images')
    save_cifar10_batches_as_images(batch_files, output_dir)