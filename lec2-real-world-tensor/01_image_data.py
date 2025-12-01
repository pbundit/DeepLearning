import imageio.v3 as iio
import matplotlib.pyplot as plt
import os
import torch


def load_and_display_image():
    image_array = iio.imread('./data/2d_data/cat1.png')
    print(image_array.shape)
    plt.imshow(image_array)
    plt.waitforbuttonpress()


def create_batch_of_images():
    batch_size = 3  # 3 images
    batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)#สร้าง tensor เปล่าที่มีแต่ค่า 0
    data_dir = 'data/2d_data/'
    filenames = [name for name in os.listdir(data_dir)
                 if os.path.splitext(name)[-1] == '.png']
    for i, filename in enumerate(filenames):
        img_array = iio.imread(os.path.join(data_dir, filename))
        img_t = torch.from_numpy(img_array)

        # note: permute operation
        img_t = img_t.permute(2, 0, 1)
        batch[i] = img_t
    print(batch.shape)
    return batch


def normalize_data(batch):
    batch = batch.float()
    batch = batch / 255.0
    return batch


def standardize_data(batch):
    n_channels = batch.shape[1]
    for c in range(n_channels):
        mean = torch.mean(batch[:, c])
        std = torch.std(batch[:, c])
        batch[:, c] = (batch[:, c] - mean)/std
    return batch


if __name__ == "__main__":
    # load_and_display_image()
    batch = create_batch_of_images()
    normalized_batch = normalize_data(batch)

    load_and_display_image()
