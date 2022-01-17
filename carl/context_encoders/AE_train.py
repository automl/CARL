from functools import partial
import os
from xvfbwrapper import Xvfb
import configargparse
import yaml
import json
import numpy as np

import sys
import inspect
import pdb

import torch as th
import pickle
from tqdm import tqdm

# Set the directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(os.getcwd())


from context_encoders import ContextAE

def  get_parser() -> configargparse.ArgumentParser:
    """
    Creates new argument parser for running baselines.

    Returns
    -------
    parser : argparse.ArgumentParser

    """
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.ConfigparserConfigFileParser
    )
    

    parser.add_argument(
        "--outdir", 
        type=str, 
        default="tmp/test_logs", 
        help="Output directory"
    ) 

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of episodes to evaluate policy on",
    )

    parser.add_argument(
        "--context_db",
        type=str,
        default="../data/context_db.json",
        help="location of context database or training"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.029983722471931533,
        help="Learning rate for training"
    )

    return parser



def train_AE():
    pass

def main(args, unknown_args, parser):
    # Model Initialization
    model = ContextAE()
    
    # Validation using MSE Loss function
    loss_function = th.nn.MSELoss()
    
    # Using an Adam Optimizer with lr = 0.1
    optimizer = th.optim.Adam(
                    model.parameters(),
                    lr = args.learning_rate,
                    weight_decay = 1e-8
                )


    # Load the database 

     
    dataset = np.load(args.context_db)

    np.random.shuffle(dataset)

    train_set = dataset[:int(0.8*len(dataset))]
    val_set   = dataset[int(-0.2*len(dataset)):]


    outputs = []
    losses = []

    # Training the model
    train_loader = th.utils.data.DataLoader(dataset = train_set,
                                     batch_size = args.batch_size,
                                     shuffle = True)
    
    epochs = args.epochs

    # for epoch in tqdm(range(epochs)):
    #     for (vector) in train_loader:
                
            
    #         #print(vector)
    #         #pdb.set_trace()
    #         # Output of Autoencoder
    #         reconstructed = model(vector)
                
    #         # Calculating the loss function
    #         loss = loss_function(reconstructed, vector)
            
    #         loss = loss.float()

    #         # The gradients are set to zero,
    #         # the the gradient is computed and stored.
    #         # .step() performs parameter update
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
                
    #         # Storing the losses in a list for plotting
    #         losses.append(loss)
    #         outputs.append((epoch, vector, reconstructed))


    # print(f'Final Training Loss: {losses[-1].item()}')
    # Saving the model
    #out.mkdir(parents=True, exist_ok=True)

    val_loader = th.utils.data.DataLoader(dataset = train_set,
                                     batch_size = args.batch_size,
                                     shuffle = True)

    val_losses = []
    val_outputs = []

    for epoch in range(10):
        for (vector) in val_loader:

            # Output of Autoencoder
            val_reconstructed = model(vector)
                
            # Calculating the loss function
            val_loss = loss_function(val_reconstructed, vector)

            # Storing the losses in a list for plotting
            val_losses.append(val_loss)
            val_outputs.append((epoch, vector, val_reconstructed))


    print(f'Final Validation Loss: {val_losses[-1]}')

    # if not os.path.exists(args.outdir):
    #     os.makedirs(args.outdir)


    # with open(os.path.join(args.outdir, 'losses.pkl'), 'wb') as f:
    #     pickle.dump(losses, f)

    # with open(os.path.join(args.outdir, 'representations.pkl'), 'wb') as f:
    #     pickle.dump(model.get_representation(), f)

    # with open(os.path.join(args.outdir, 'opt.pkl'), 'wb') as f:
    #     pickle.dump(optimizer.state_dict(), f)


    # th.save(model.state_dict(), os.path.join(args.outdir, 'model.zip'))



if __name__ == '__main__':
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    main(args, unknown_args, parser)