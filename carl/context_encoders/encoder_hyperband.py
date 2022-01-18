import torch as th
import numpy as np
import os
import pdb
from tqdm import tqdm
import pickle
from mle_hyperopt import HyperbandSearch

from carl.context_encoders import ContextEncoder, ContextAE, ContextVAE, ContextBVAE
import json
import shutil

import hydra
from omegaconf import DictConfig

step = 0
base_dir = os.getcwd()


@hydra.main("./configs", "base")
def main(cfg: DictConfig) -> None:

    global base_dir

    out_dir = os.path.join(base_dir, cfg.encoder.outdir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not cfg.encoder.context_dataset:
        raise ValueError(
            "Please specify a context dataset. To generate a context dataset, use the 'generate_context_dataset.py' on a contexts_train.json sampled by running an environment with required specifications"
        )

    dataset = np.load(os.path.join(base_dir, cfg.encoder.context_dataset))

    np.random.shuffle(dataset)
    train_set = dataset[: int(0.8 * len(dataset))]
    val_set = dataset[int(-0.2 * len(dataset)) :]

    # Blackbox objective
    def train_model(model, **kwargs):

        global step

        for key in kwargs:
            print(key, kwargs[key])
            print("\t")

        iter_dir = os.path.join(out_dir, f"iter_{step}")
        if not os.path.exists(iter_dir):
            os.mkdir(os.path.join(iter_dir))
        else:
            shutil.rmtree(os.path.join(iter_dir))
            os.mkdir(os.path.join(iter_dir))

        optimizer = th.optim.Adam(
            model.parameters(), lr=kwargs["rate"], weight_decay=kwargs["decay"]
        )

        loader = th.utils.data.DataLoader(
            dataset=train_set, batch_size=kwargs["batch"], shuffle=False
        )

        epochs = cfg.encoder.epochs
        losses = []
        for _ in tqdm(range(epochs)):

            for vector in loader:

                # Output of Autoencoder
                results = model(vector)

                # Get the loss
                loss_dict = model.loss_function(*results, M_N=cfg.encoder.M_N)
                loss = loss_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Storing the losses in a list for plotting
                losses.append(loss.item())

        print(f"Final Training Loss: {losses[-1]}")

        ## Validation
        loader = th.utils.data.DataLoader(
            dataset=val_set, batch_size=kwargs["batch"], shuffle=False
        )

        val_losses = []
        for _ in range(10):
            for vector in loader:

                # Output of Autoencoder
                val_results = model(vector)

                # Get hte validation loss
                val_loss_dict = model.loss_function(*val_results, M_N=cfg.encoder.M_N)
                val_loss = val_loss_dict["loss"]

                # Storing the losses in a list for plotting
                val_losses.append(val_loss.item())

        print(f"Mean Validation Loss: {np.mean(val_losses)}")

        ## Save the stuff
        np.save(os.path.join(iter_dir, "losses.npy"), losses)

        with open(os.path.join(iter_dir, "representations.pkl"), "wb") as f:
            pickle.dump(model.get_representation(), f)

        with open(os.path.join(iter_dir, "opt.pkl"), "wb") as f:
            pickle.dump(optimizer.state_dict(), f)

        th.save(model, os.path.join(iter_dir, "model.zip"))

        step = step + 1

    # Create a sampling strategy
    strategy = HyperbandSearch(
        real={
            "rate": {"begin": 0.001, "end": 0.5, "prior": "uniform"},
            "decay": {"begin": 1e-16, "end": 1e-4, "prior": "uniform"},
        },
        integer={"batch": {"begin": 30, "end": 128, "prior": "log-uniform"}},
        search_config={"max_resource": 14, "eta": 3},
        # seed=42, # Fix the order of sampled configs if comapring dfferent runs
    )

    # Generate the configs using the strategy and dump them
    configs = strategy.ask()
    with open(os.path.join(out_dir, "configs.json"), "w") as f:
        json.dump(configs, f, indent=4)

    encoder_cls = eval(cfg.encoder.model)

    model = encoder_cls(
        cfg.encoder.input_dim, cfg.encoder.latent_dim, cfg.encoder.hidden_dim
    )

    # Train the model with the generated configs
    for c in configs:
        train_model(model, **c["params"])


if __name__ == "__main__":
    main()
