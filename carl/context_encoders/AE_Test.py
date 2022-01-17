import torch as th
import numpy as np
from functools import partial
import os
import sys
import inspect
import pdb
from tqdm import tqdm
import pickle
from mle_hyperopt import HyperbandSearch

from context_encoders import ContextAE
import pprint
import json


out_dir = os.path.join(os.getcwd(), 'tmp/AE_stuff/AE_Tune_WD_norm_1')

# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)

context_db = os.path.join (os.getcwd(), 'tmp', 'AE_stuff/AE_database','new_60k.npy')

dataset = np.load(context_db)

# Normalize data
mins  = np.min(dataset, axis = 0)
maxs   = np.max(dataset, axis = 0)

for idx in range(dataset.shape[1]):
    print(len(dataset[:,idx]) )
    dataset[:,idx] = (dataset[:,idx] - mins[idx]) /(maxs[idx] - mins[idx])

np.random.shuffle(dataset)

np.random.shuffle(dataset)

val_set = dataset[int(-0.2*len(dataset)):]

val_set   = [
                dataset[
                    int(0.2*i*len(dataset))
                    :int(0.2*(i+1)*len(dataset))
                ] for i in range(5) 
            ]

# Blackbox objective
def test_model(
    model: ContextAE(5,2),
    batch: int):

    loss_function = th.nn.MSELoss()
    epochs = 10

    main_val_loss = []

    #for val in val_set:
    
    ## Validation 

    for val in val_set:
        loader = th.utils.data.DataLoader(dataset = val,
                                        batch_size = batch,
                                        shuffle = False)
            
        val_losses = []
        for epoch in range(epochs):
            for (vector) in loader:

                # Output of Autoencoder
                val_reconstructed = model(vector)
                    
                # Calculating the loss function
                val_loss = loss_function(val_reconstructed, vector)

                # Storing the losses in a list for plotting
                val_losses.append(val_loss.item())


        main_val_loss.append(np.mean(val_losses))

    return main_val_loss


with open(os.path.join(os.getcwd(), 'tmp/AE_stuff/AE_Tune_WD_norm_1', 'configs.json'), 'r') as f:
    configs = json.loads(f.read())


for i in range(9):
    batch = configs[i]['params']['batch']
    rate = configs[i]['params']['rate']
    decay = configs[i]['params']['decay']

    model = ContextAE(5,1)
    model.load_state_dict(th.load(os.path.join(out_dir, f'iter_{i}', 'model.zip')))
    # model = th.load(os.path.join(out_dir, f'iter_{i}', 'model.zip'))

    #rint(type(model))


    print(f'Batch_Size: {batch}    LR: {rate}    Decay: {decay}    Val_losses: {test_model(model, 1)}')

