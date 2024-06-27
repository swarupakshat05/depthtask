import OpenEXR
import numpy as np
import Imath
import matplotlib.pyplot as plt
import cv2

exr_file_path = "/content/00003Left.exr"

def load_exr(file_path, channel_name):
    exr = OpenEXR.InputFile(file_path)
    # Get the header to obtain the data window
    header = exr.header()
    data_window = header['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_data = exr.channel(channel_name, pixel_type)

    # Convert the data to numpy array
    depth_data = np.frombuffer(channel_data, dtype=np.float32)
    depth_data.shape = (size[1], size[0])  # (height, width)

    return depth_data

def visualize_depth_map(depth_data, channel_label):
    plt.imshow(depth_data, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title(f'{channel_label} Depth Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Load the depth map from EXR file
depth_map_r = load_exr(exr_file_path, 'R')
depth_map_g = load_exr(exr_file_path, 'G')
depth_map_b = load_exr(exr_file_path, 'B')

visualize_depth_map(depth_map_r, 'R')
visualize_depth_map(depth_map_g, 'G')
visualize_depth_map(depth_map_b, 'B')

combined_depth_map = np.stack([depth_map_r, depth_map_g, depth_map_b], axis=-1)
combined_depth_map = cv2.cvtColor(combined_depth_map, cv2.COLOR_RGB2BGR)
grayscale_depth_map = cv2.cvtColor(combined_depth_map, cv2.COLOR_BGR2GRAY)

# Visualize the combined grayscale depth map
visualize_depth_map(grayscale_depth_map, 'RGB')

def show_histogram(image):
    flattened_pixels = image.flatten()
    plt.hist(flattened_pixels, bins=256, range=(0, 256), density=True, color='blue', alpha=0.7)
    plt.title('Histogram of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

show_histogram(grayscale_depth_map.copy())
