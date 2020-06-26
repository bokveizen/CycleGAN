import matplotlib.pyplot as plt
from pathlib import Path
import torch
import cv2
import numpy as np

attn_maps = Path('attn_h2z')
attn_maps_files = list(attn_maps.iterdir())
for f in attn_maps_files:
    attn_maps_ex = torch.load(f, map_location=torch.device('cpu'))
    img_array = 0.5 * (attn_maps_ex[0][0] + 1).permute(1, 2, 0).numpy()
    plt.subplot(221)
    plt.axis('off')
    plt.imshow(img_array)
    # plt.show()
    attn_maps_tensor = attn_maps_ex[1][0]
    attn_maps_flatten = torch.zeros(attn_maps_tensor.shape[0])
    for i in range(attn_maps_tensor.shape[0]):
        attn_maps_flatten[i] = attn_maps_tensor[i].mean()
    # attn_maps_flatten /= attn_maps_flatten.max()
    _, top_2_index = attn_maps_flatten.topk(2)
    for i in top_2_index:
        attn_maps_flatten[i] = 1.
    attn_maps_flatten = attn_maps_flatten.reshape(int(attn_maps_tensor.shape[0] ** 0.5),
                                                  int(attn_maps_tensor.shape[0] ** 0.5))
    plt.subplot(222)
    plt.axis('off')
    plt.imshow(attn_maps_flatten, cmap='gray')
    attn_map_1 = attn_maps_tensor[top_2_index[0]].reshape(int(attn_maps_tensor.shape[0] ** 0.5),
                                                          int(attn_maps_tensor.shape[0] ** 0.5))
    attn_map_2 = attn_maps_tensor[top_2_index[1]].reshape(int(attn_maps_tensor.shape[0] ** 0.5),
                                                          int(attn_maps_tensor.shape[0] ** 0.5))
    plt.subplot(223)
    plt.axis('off')
    plt.imshow(attn_map_1, cmap='gray')

    plt.subplot(224)
    plt.axis('off')
    plt.imshow(attn_map_2, cmap='gray')

    plt.show()
