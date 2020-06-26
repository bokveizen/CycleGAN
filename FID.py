import torch
import torch.nn as nn
import numpy as np
from metric.inception import InceptionV3
from scipy.linalg import sqrtm
import argparse
import os
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='the name of the result folder')
args = parser.parse_args()
result_folder = 'results/{}/test_latest/images'.format(args.name)


## Step 3 : Your Implementation Here ##
## Implement functions for fid score measurement using InceptionV3 network ##
def fid(m_r, m_f, C_r, C_f):
    # fanchen: google python matrix square root --> scipy.linalg.sqrtm
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
    m_r, m_f = np.atleast_1d(m_r), np.atleast_1d(m_f)
    C_r, C_f = np.atleast_2d(C_r), np.atleast_2d(C_f)

    term_1 = np.dot(m_r - m_f, m_r - m_f)
    term_2 = np.trace(C_r + C_f)
    CCsquare = sqrtm(np.dot(C_f, C_r))
    term_3 = -2 * np.trace(CCsquare)

    return term_1 + term_2 + term_3


def mean_var_cal(img_folder_path, tag, model):
    files = [x for x in Path(img_folder_path).iterdir() if tag in x.name]
    # act = get_activations(files, model, 10, 2048, verbose)
    model.eval()
    act = np.empty((len(files), 2048))
    for i in range(0, len(files), 10):
        images = np.array([np.asarray(Image.open(f), dtype=np.uint8)[..., :3].astype(np.float32)
                           for f in files[i:i + 10]])
        images = images.transpose((0, 3, 1, 2))  # N,C,H,W
        images /= 255  # 0-1 norm
        batch = torch.from_numpy(images).type(torch.FloatTensor)
        # batch.cuda()
        pred = model(batch)[0]
        act[i:i + 10] = pred.cpu().data.numpy().reshape(pred.size(0), -1)
    print('mean_val_cal done for tag', tag)
    m, C = np.mean(act, axis=0), np.cov(act, rowvar=False)
    return m, C


def main(img_folder_path):
    inception_model = InceptionV3()
    # inception_model.cuda()

    m_r_A, C_r_A = mean_var_cal(img_folder_path, 'real_A', inception_model)
    m_f_A, C_f_A = mean_var_cal(img_folder_path, 'fake_A', inception_model)
    m_r_B, C_r_B = mean_var_cal(img_folder_path, 'real_B', inception_model)
    m_f_B, C_f_B = mean_var_cal(img_folder_path, 'fake_B', inception_model)

    fid_value_A = fid(m_r_A, m_f_A, C_r_A, C_f_A)
    fid_value_B = fid(m_r_B, m_f_B, C_r_B, C_f_B)
    print(args.name, 'fid_value_A', fid_value_A, 'fid_value_B', fid_value_B)
    return fid_value_A, fid_value_B


if __name__ == '__main__':
    main(result_folder)
