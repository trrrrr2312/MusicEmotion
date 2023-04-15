import numpy as np
import torch
from tqdm import tqdm
from utils.config import config


def torch_to_cv2(images):
    """
    Converts a collection of PyTorch images generated by a GAN to cv2-compatible format
    :param images:
    :return:
    """
    for i in range(len(images)):
        cur_imgs = images[i]
        cur_imgs = cur_imgs.transpose((0, 2, 3, 1))[:, :, :, ::-1].copy()
        cur_imgs = np.uint8(np.clip(((cur_imgs + 1) / 2.0) * 256, 0, 255))
        images[i] = cur_imgs
    images = np.concatenate(images)
    return images


def feedforward_gan(model, class_vectors, noise_vectors, batch_size, truncation):
    """
    Feedd-fowards the GAN and creates a collection of images
    :param model:
    :param class_vectors:
    :param noise_vectors:
    :param batch_size:
    :param class_ids:
    :param truncation:
    :return:
    """

    images = []
    n_batches = int(len(class_vectors) / batch_size)
    print("n_batches")
    print(n_batches)
    print("Generating GAN content...")
    for i in tqdm(range(n_batches)):
        cur_noise = torch.from_numpy(noise_vectors[i * batch_size:(i + 1) * batch_size]).to(config['device'])
        cur_class = torch.from_numpy(class_vectors[i * batch_size:(i + 1) * batch_size]).to(config['device'])

        with torch.no_grad():
            output = model(cur_noise, cur_class, truncation)

        images.append(output.cpu().numpy())

    if n_batches * batch_size < len(class_vectors):
        cur_noise = torch.from_numpy(noise_vectors[n_batches * batch_size:]).to(config['device'])
        cur_class = torch.from_numpy(class_vectors[n_batches * batch_size:]).to(config['device'])

        with torch.no_grad():
            output = model(cur_noise, cur_class, truncation)

        images.append(output.cpu().numpy())
    images = torch_to_cv2(images)
    return images
