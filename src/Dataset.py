import torch
import sys
sys.path.insert(0,'../../primate-cnn-model')
import src.config as cfg
import src.misc_util as mu
import numpy as np
import os

class Dataset():
    
    """
    Dataset class for mini-batch gradient descent.
    
    @param datapath: path to the files for training/testing
    @param partition: either 'train','test', or 'valid'
    @param history_frames: number of history frames to use in the stimulus
    @param cell_inds: indices of cells to use for training
    @return: object with X,Y pairs for use with model in PyTorch
    
    @author Alex Gogliettino and Sam Cooler
    """
    
    def __init__(self,datapath,partition,history_frames,cell_inds):
        self.history_frames = history_frames

        if partition in ['train']:
            self.X = np.load(os.path.join(
                                    datapath,cfg.TRAIN_X_FNAME)
                            ).swapaxes(1,2)
            self.Y = np.load(os.path.join(
                                    datapath,cfg.TRAIN_Y_FNAME)
                     )[:,cell_inds]
        elif partition in ['test']:
            self.X = np.load(os.path.join(
                                    datapath,cfg.TEST_X_FNAME)
                    ).swapaxes(1,2)
            self.Y = np.load(os.path.join(
                                    datapath,cfg.TEST_Y_FNAME)
                        )[:,cell_inds]
        elif partition in ['valid']:
            self.X = np.load(os.path.join(
                                    datapath,cfg.VALID_X_FNAME)
                    ).swapaxes(1,2)
            self.Y = np.load(os.path.join(
                                    datapath,cfg.VALID_Y_FNAME)
                        )[:,cell_inds]
        else:
            assert False

        # Rescale the image and mean subtract.
        self.X = self.X / cfg.MONITOR_BIT_DEPTH
        self.X -= np.mean(self.X)

    def roll_and_crop(self):
        """
        Sets the first non-batch dimension as stimulus history, and also
        truncates the response data by removing those time points.
        """
        self.X = mu.rolling_window(self.X,self.history_frames)
        self.Y = self.Y[self.history_frames:,:]
        
    def __getitem__(self,index):
        x = self.X[index,...]
        y = self.Y[index,...]

        return x,y
    
    def __len__(self):
        return self.X.shape[0]