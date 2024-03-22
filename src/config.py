"""
Configuration file with constants, model hyperparameters and other things.

Some things don't change, others feel free to depending on what sort of model
you want to train.

@author Alex Gogliettino and Sam Cooler
"""

########### Below are what you might want to change for each model ############

""" Model architecture/data specifics """

"""
history_frames: number of frames to use from present time in the model fitting.
As long as it exceeds the STA duration, it's fine. In our case, 50 frames
at 8.33 ms refresh is more than adequate
"""
history_frames = 50

'''
Flags to enable/disable the first two convolutional layers. If you want 3-layer 
CNN, both need to be True. If you want an LN model, both need to be False. This
example thus trains a 3 layer CNN.
'''
enable_layer_0 = True  
enable_layer_1 = True

"""
Number of convolutional kernels for each layer. 8/16 was found to work well 
for parasols and midget cells but if you have different cell types, you might
need to change this up.
"""
conv0_channels = 8
conv1_channels = 16

'''
This kernel size is fine for parasols/midgets, based on RF size on a grid of
80x160. Will heavily depend on the size of the grid you're using
and if you try other cell types with bigger RF sizes, probably have to
change.
'''
conv0_kernel_size = 16
conv1_kernel_size = 16

'''
Output nonlinearity for fully-connected layer. At the moment, no other
option is supported, but could play around with other ones if you 
wanted.
'''
nonlinearity = 'softplus'

"""
For the CNN with the above hyperparameters, a grid search yielded these
as the best. Could maybe do a bit more exploration. If using additional
cell types, probably want to explore more. 

For the LN model, it is HIGHLY SENSITIVE to regularization; because the
LN model has a ton of parameters per cell, it overfits easily. I found
LN model to do best with linear_L1_reg_lamda = 1e-7 for parasol/midget.

Because when training a LN model, the first two layers are disabled so the
penalty weight doesn't matter for conv0/conv1.
""" 
conv0_L2_reg_lamda = 1e-5 # for CNN
conv1_L2_reg_lamda = 1e-5 # for CNN
linear_L1_reg_lamda = 1e-5 # for CNN; 1e-7 for LN.

"""
Gaussian noise injected during training in layers 1/2. Helps stabilize training. 
I didn't explore this much but certainly could play more with it. These numbers
are standard deviation of a standard normal distribution.
"""
layer_0_noise = 0.01
layer_1_noise = 0.01


""" Training specifics """

""" Which GPU to use."""
device = 'cuda:0'

""" Learning rate: 1e-4 worked fine, but can play with it. Might also consider
a learning rate scheduler for better training, but I did not explore this.
"""
learning_rate = 1e-4

"""
Number of training epochs. In these data with the hyperparameters listed here,
5-10 for LN and 10-15 for CNN, but always check validation error.
"""
epochs = 10

"""
Batch size for mini-batch gradient descent. 32 is probably fine (depending on
your GPU), but it's always recommended to have a power of 2 for maximum 
performance, if you want to modify.
"""
batch_size = 32

"""
Number of batches after which to print training loss to the console.
"""
N_BATCHES_TO_PRINT = 100

###############################################################################


""" The rest you only need to change if you have different file names."""

# 8 bit monitor precision: DO NOT MODIFY unless your stimulus is not 8-bit.
MONITOR_BIT_DEPTH = 2**8 - 1 

"""
Where the cell indices by type file is written to. Here, we have a Python
dictionary written as a .npy file which maps cell type (string) to a NumPy 
array of integers, which correspond to the indices of the cells of the 
corresponding type.
"""
CELLIDS_DICT_FNAME = 'cell_inds_by_type.npy'

"""
Train/validation/test partition file names. X is stimulus, Y is firing rate.
X has dimension frames x 80 x 160, Y has dimension frames x cells, whose 
indexing is consistent with the cell indices in CELLIDS_DICT_FNAME.
"""
TRAIN_X_FNAME = 'train_X.npy' 
TRAIN_Y_FNAME = 'train_Y.npy'
VALID_X_FNAME = 'valid_X.npy'
VALID_Y_FNAME = 'valid_Y.npy'
TEST_X_FNAME = 'test_X.npy'
TEST_Y_FNAME = 'test_Y.npy'

"""
File name for the model outputs, saved after each epoch.
"""
MODEL_OUT_FNAME = 'cnn_out_epoch_%s.tar'