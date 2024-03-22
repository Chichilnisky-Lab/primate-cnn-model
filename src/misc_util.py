import sys
sys.path.insert(0,'../../primate-cnn-model')
import src.config as cfg
import numpy as np
import torch
from torch.nn import functional as F
from typing import *

"""
Miscellaneous utility functions used in model definition and dataset class.

@author Alex Gogliettino and Sam Cooler.
"""

def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension
    Parameters
    From: https://github.com/baccuslab/deep-retina/tree/master/deepretina 
    ----------
    array : array_like
        Array to add rolling window to
    window : int
        Size of rolling window
    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.
    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])
    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if time_axis == 0:
        array = array.T

    elif time_axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr

def get_conv_output_shape(input_dims,kernel_size,stride=1,pad=0):
    """
    Gets 2D convolution output shape from input/kernel size, stride and 
    padding
    @param input_dims: dimension of input (x,y), int
    @param kernel_size: size of kernel (int)
    @param stride: stride of the convolution (int)
    @param pad: extent of padding (int)
    @return tuple (x,y) of the resulting convolution output
    
    @author Alex Gogliettino and Sam Cooler
    """
    y,x = input_dims
    out_y = ((y + (2 * pad) - kernel_size) // stride) + 1
    out_x = ((x + (2 * pad) - kernel_size) // stride) + 1

    return out_y,out_x