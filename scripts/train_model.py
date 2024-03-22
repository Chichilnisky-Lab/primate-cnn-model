import sys
sys.path.insert(0,'../../primate-cnn-model/')
import numpy as np
import os
import src.config as cfg
import src.models as models
import src.Dataset as d
import torch
from tqdm import tqdm

"""
A training script to train a CNN/LN model on primate RGCs responses to 
stimulation with naturalistic images.

Sets up a model config that gets sent to the model and initiates a training
loop.

Here, just set the path to the training data and the path you want to write
the model files too, which get written after every training epoch.

Please refer to config.py to change hyperparameters of the model and other
settings.

@author Alex Gogliettino and Sam Cooler
"""

####################### Edit below for each model. ############################

# Set the path to where the data live and load in the cell indices by type.
dirin = '../data'
cell_inds_by_type = np.load(os.path.join(dirin,cfg.CELLIDS_DICT_FNAME),
                            allow_pickle=True
                    ).item()

"""
Specify a checkpoint file if you want to resume training a previous model. 
Empty string will generate a new model from scratch. If you specify a file 
name with epoch N and you set epochs to M, it will still train M more epochs.
"""
checkpoint_file = '' 

 # Set the output directory for model files to be written to.
dirout = os.path.join(dirin,'cnn-out')

"""
These are the indices that you actually want for model training.
In this example, all cells will be used for training, but you can select any 
indices.
"""
cell_inds = np.sort(np.r_[
                        cell_inds_by_type['on parasol'],
                        cell_inds_by_type['off parasol'],
                        cell_inds_by_type['on midget'],
                        cell_inds_by_type['off midget']
                    ])

####################### Edit above for each model. ############################

if not os.path.isdir(dirout):
    os.makedirs(dirout)

train_data = d.Dataset(dirin,'train',cfg.history_frames,cell_inds)
test_data = d.Dataset(dirin,'valid',cfg.history_frames,cell_inds)
train_data.roll_and_crop()
test_data.roll_and_crop()
y_dim,x_dim = train_data.X.shape[2:4]

# Write model configuration which gets passed to the model class.
model_config = {
    'history_frames': cfg.history_frames, 'nonlinearity': cfg.nonlinearity,
    'enable_layer_0': cfg.enable_layer_0,'enable_layer_1': cfg.enable_layer_1,
    'layer_0': {'channels': cfg.conv0_channels, 
                'kernel_size': cfg.conv0_kernel_size
                },
    'layer_1': {'channels': cfg.conv1_channels,
                    'kernel_size': cfg.conv1_kernel_size
    },
    'stimulus_dim': (y_dim,x_dim), 'n_cells': cell_inds.shape[0],
    'learning_rate': cfg.learning_rate,
    'conv0_L2_reg_lamda': cfg.conv0_L2_reg_lamda,
    'conv1_L2_reg_lamda': cfg.conv1_L2_reg_lamda,
    'linear_L1_reg_lamda': cfg.linear_L1_reg_lamda,
    'layer_0_noise': cfg.layer_0_noise,'layer_1_noise': cfg.layer_1_noise,
    'batch_size': cfg.batch_size,
    'output_scale_initialize': np.max(train_data.Y,
                                        axis=0),
    'device': cfg.device
}

Model = models.CNN(model_config)
start_epoch = 0

# Check if a checkpoint is desired.
if checkpoint_file:
    fnamein = os.path.join(dirout,checkpoint_file)

    assert os.path.isfile(fnamein)

    model_dict = torch.load(fnamein,map_location='cpu')
    Model.load_state_dict(model_dict['model_state_dict'])
    start_epoch = int(
                    checkpoint_file.split('epoch_')[-1].split('.tar')[0]
                ) + 1

Model = Model.to(cfg.device)

# Default to Poisson loss, but could also try MSE.
criterion = torch.nn.PoissonNLLLoss(log_input=False, full=True)
optimizer = optimizer = torch.optim.Adam(
                            Model.parameters(), 
                            lr=model_config['learning_rate']
                        )
Model.train()

for epoch in tqdm(
                np.arange(start_epoch,start_epoch+cfg.epochs),
                ascii=True,desc='Epochs'
            ):
    train_cost_cache = []
    test_cost_cache = []
    reg_cost_cache = []
    train_loader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=cfg.batch_size,
                        shuffle=True
            )
    test_loader = torch.utils.data.DataLoader(
                                test_data,
                                batch_size=cfg.batch_size,
                                shuffle=True
                )
    
    for i, (X,Y) in enumerate(train_loader,0):
        X = torch.tensor(X).to(torch.float32).to(cfg.device)
        Y = torch.tensor(Y).to(torch.float32).to(cfg.device)
        
        """ 
        Sometimes produces negatives, which doesn't obviously work with Poisson 
        loss so have to rectify (plus spikes can't be < 0).
        """
        Y_hat = torch.clamp(Model(X),min=0)
        loss = criterion(Y_hat,Y)

        # Do L2 regularization for first two layers and add it to the cost.
        if not cfg.enable_layer_0:
            reg_conv0 = 0
        else:
            reg_conv0 =\
                cfg.conv0_L2_reg_lamda *\
                    sum(param.pow(2.0).sum() 
                    for name, param in Model.named_parameters()
                    if 'conv0.weight' == name)
                    
        if not cfg.enable_layer_1:
            reg_conv1 = 0
        else:
            reg_conv1 =\
                cfg.conv1_L2_reg_lamda *\
                    sum(param.pow(2.0).sum() 
                    for name, param in Model.named_parameters()
                    if 'conv1.weight' == name)
                    
        # Default to L1 for linear regardless of model type
        reg_linear =\
            cfg.linear_L1_reg_lamda *\
                sum(param.abs().sum() 
                for name, param in Model.named_parameters() 
                if 'linear.weight' == name)

        # Sum the penalty terms with the training loss for regularization.
        reg = reg_conv0 + reg_conv1 + reg_linear
        loss += reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cost = loss.item()
        train_cost_cache.append(cost)
        reg_cost_cache.append(reg)

        if i % cfg.N_BATCHES_TO_PRINT == 0:
            print(cost)
    
    # Evaluate the model on the validation data. 
    Model.eval()

    for _, (X_test,Y_test) in enumerate(test_loader,0):
        X_test = torch.tensor(X_test).to(torch.float32).to(cfg.device)
        Y_test = torch.tensor(Y_test).to(torch.float32).to(cfg.device)
        Y_hat_test = torch.clamp(Model(X_test),min=0)
        loss = criterion(Y_hat_test,Y_test)
        test_cost_cache.append(loss.item())
    
    Model.train()
    
    fnameout = os.path.join(dirout,cfg.MODEL_OUT_FNAME%str(epoch))
    torch.save({
        'model_config': model_config,
        'epoch': epoch,'train_cost_cache': train_cost_cache,
        'test_cost_cache': test_cost_cache,
        'reg_cost_cache': reg_cost_cache,
        'model_state_dict': Model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    },fnameout)