
import numpy as np
from scipy.stats import entropy
import torch
import torch.nn as nn
import os

def meanRound(arr:list, digits:int):
    return round(np.mean(arr),digits)
