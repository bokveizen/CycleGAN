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
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(img_array)
    # plt.show()
    attn_maps_tensor = attn_maps_ex[1][0]
    attn_maps_vis_size = 256
    ratio = attn_maps_tensor.shape[0] // attn_maps_vis_size
    attn_maps_array = np.zeros((attn_maps_vis_size, attn_maps_vis_size))
    for i in range(attn_maps_array.shape[0]):
        for j in range(attn_maps_array.shape[1]):
            attn_maps_array[i][j] = float(attn_maps_tensor[i * ratio:(i + 1) * ratio,
                                          j * ratio:(j + 1) * ratio].max().numpy())
    # plt.imshow(cv2.cvtColor(attn_maps_ex[0][0], cv2.COLOR_BGR2RGB))
    # plt.imshow(attn_maps_ex[0][0])
    # plt.show()
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(attn_maps_array, cmap='gray')
    plt.show()
