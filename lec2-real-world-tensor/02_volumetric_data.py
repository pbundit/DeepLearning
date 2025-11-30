import imageio
import matplotlib.pyplot as plt
import torch


def load_volumetric_data():
    file_path = "data/3d_data"
    vol_array = imageio.volread(file_path, 'DICOM')
    print(vol_array.shape)
    return vol_array


def visualize_slice(vol_array):
    slice = vol_array[10]  # take one of the slice
    plt.imshow(slice)
    plt.waitforbuttonpress()


def create_tensor(vol_array):
    vol_tensor = torch.tensor(vol_array)
    return vol_tensor


if __name__ == "__main__":
    vol_array = load_volumetric_data()
    visualize_slice(vol_array)
