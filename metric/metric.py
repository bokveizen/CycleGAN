import torch
import torch.nn as nn
import numpy as np
from metric.inception import InceptionV3
from scipy.linalg import sqrtm


## Step 3 : Your Implementation Here ##
## Implement functions for fid score measurement using InceptionV3 network ##

def fid(m_f, m_r, C_f, C_r):
    # fanchen: google python matrix square root --> scipy.linalg.sqrtm
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
    CCsquare = sqrtm(np.dot(C_f, C_r))

