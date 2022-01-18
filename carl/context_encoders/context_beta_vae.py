import torch as th
from typing import List

from carl.context_encoders.context_encoder import ContextEncoder


class ContextBVAE(ContextEncoder):
    """
    Implementation of a Beta-Variational Autoencoder (https://openreview.net/forum?id=Sy2fzU9gl) 
    that learns to reconstruct a context vector, while optimizing for a factorized distribution 
    in the latent space, using an adjustable hyperparameter beta that balances latent channel 
    capacity and independence constraints with reconstruction accuracy

    Structure adapted from: https://github.com/AntixK/PyTorch-VAE

    Parameters
    ----------
    input_dim: int
        Dimensions of the context vector being fed to the Autoencoder
    latent_dim: int
        Dimensions of the latent representation of the context vector
    hidden_dims: List[int]
        List of hidden dimensions to be used by the encoder and decoder
    beta: int
        Beta hyperparameter for the beta distribution
    gamma: float
        Gamma hyperparameter for the beta distribution
    max_capacity: int
        Maximum capacity of the latent channel
    Capacity_max_iter: int
        Maximum number of iterations to reach the maximum capacity
    loss_type: str
        Loss type to be used for the beta distribution
    

    Attributes
    ----------
    encoder: th.nn.Module
        Encoder network
    decoder: th.nn.Module
        Decoder network
    representations: th.Tensor
        Latent representation of the context vector
    """
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 1,
        hidden_dims: List = [3],
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "B",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self.beta = beta
        self.gamma = gamma
        self.C_max = th.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.loss_type = loss_type

        self._build_network()

        # Registering
        self.representations = None
        self.double()

    def _build_network(self):
        """
        Builds the network
        """
        # Make the Encoder
        modules = []

        hidden_dims = self.hidden_dims
        input_dim = self.input_dim

        for h_dim in hidden_dims:
            modules.append(th.nn.Linear(input_dim, h_dim))
            modules.append(th.nn.ReLU())
            input_dim = h_dim

        self.encoder = th.nn.Sequential(*modules)

        # Mean and std_dev for the latent distribution
        self.fc_mu = th.nn.Linear(hidden_dims[-1], self.latent_dim)
        self.fc_var = th.nn.Linear(hidden_dims[-1], self.latent_dim)

        # Make the decoder
        modules = []

        # self.decoder_input = th.nn.Linear(self.latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        input_dim = self.latent_dim

        for h_dim in hidden_dims:
            modules.append(th.nn.Linear(input_dim, h_dim))
            modules.append(th.nn.ReLU())
            input_dim = h_dim

        modules.append(th.nn.Linear(input_dim, self.input_dim))
        modules.append(th.nn.ReLU())

        self.decoder = th.nn.Sequential(*modules)

    def forward(self, x, **kwargs):
        """
        Forward pass of the network
        """

        # Get mean and sigma for the latent distribution
        mu, log_var = self.encode(x)

        # sample the latent vector from this distribution
        z = self.reparameterize(mu, log_var)

        # Save the sampled latent vector for future recon
        self.representations = z

        # Return the recontruction, input, mean and variance
        return [self.decode(z), x, mu, log_var]

    def reparameterize(self, mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from

        """
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return eps * std + mu

    def encode(self, x) -> List[th.Tensor]:
        """
        Pass the tensor through the encoder network
        """
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, x) -> th.Tensor:
        """
        Pass the tensor through the decoder network
        """
        return self.decoder(x)

    def sample(self, num_samples: int, current_device: int = 0, **kwargs) -> th.Tensor:
        """
        Sample from the latent-space distribution

        Parameters
        ----------
        num_samples: int
            Number of samples to be drawn from the latent-space distribution
        current_device: int
            Device to be used for the sampling
        
        Returns
        -------
        th.Tensor
            Samples from the latent-space distribution
        
        """
        z = th.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def get_representation(self, context: th.Tensor) -> th.Tensor:
        """
        Get the recorded latent representations of a passed context vector
        """
        # Get mean and sigma for the latent distribution
        mu, log_var = self.encode(context)

        # sample the latent vector from this distribution
        z = self.reparameterize(mu, log_var)
        
        return z

    def get_encoder(self):
        """
        Get the encoder module
        """
        return self.encoder

    def get_decoder(self):
        """
        Get the decoder module
        """
        return self.decoder

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Loss function for the network

        Parameters
        ----------
        *args:
            Arguments to be passed to the loss function
        **kwargs:
            Keyword arguments to be passed to the loss function
        """

        self.num_iter += 1
        recons = args[0]
        ip = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        recons_loss = th.nn.functional.mse_loss(recons, ip)

        kld_loss = th.mean(
            -0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        if self.loss_type == "H":
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":
            self.C_max = self.C_max.to(ip.device)
            C = th.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {"loss": loss, "recon_loss": recons_loss, "kl_loss": kld_loss}
