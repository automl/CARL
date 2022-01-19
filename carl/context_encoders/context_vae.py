import torch as th
from typing import List

from carl.context_encoders.context_encoder import ContextEncoder


class ContextVAE(ContextEncoder):
    """
    Implementation of a Variational Autoencoder (https://arxiv.org/abs/1312.6114) that
    learns to reconstruct a context vector.

    Structure adapted from: https://github.com/AntixK/PyTorch-VAE

    Parameters
    ----------
    input_dim: int
        Dimensions of the context vector being fed to the Autoencoder
    latent_dim: int
        Dimensions of the latent representation of the context vector
    hidden_dims: List[int]
        List of hidden dimensions to be used by the encoder and decoder

    Attributes
    ----------
    encoder: th.nn.Module
        Encoder network
    decoder: th.nn.Module
        Decoder network
    representations: th.Tensor
        Latent representation of the context vector
    """

    def __init__(
        self, input_dim: int = 5, latent_dim: int = 1, hidden_dims: List = [3]
    ):

        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self._build_network()

        # Registering
        self.representations = None
        # self.encoder.register_forward_hook(self._representation_hook)
        self.double()

    def _build_network(self):
        """
        Builds the encoder and decoder networks
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
        Forward pass of the VAE
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
        Reparameterization trick to sample from N(mu, var)
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
        Sample from the VAE

        Parameters
        ----------
        num_samples: int
            Number of samples to be drawn
        current_device: int
            Device to be used for sampling

        Returns
        -------
        th.Tensor
            Samples from the VAE
        """
        z = th.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def get_representation(self, context: th.Tensor) -> th.Tensor:
        """
        Get the recorded latent representations of the encoder
        """
        # Get mean and sigma for the latent distribution
        mu, log_var = self.encode(context)

        # sample the latent vector from this distribution
        z = self.reparameterize(mu, log_var)

        return z

    def get_encoder(self) -> th.nn.Module:
        """
        Get the encoder module
        """
        return self.encoder

    def get_decoder(self) -> th.nn.Module:
        """
        Get the decoder module
        """
        return self.decoder

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
            KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        Parameters:
        -----------
        *args: list
            List of tensors containing the reconstruction, input, mean and variance of the VAE
        **kwargs: dict
            Dictionary of keyword arguments

        Returns:
        --------
        dict
            Dictionary containing the loss and the loss components
        """
        recons = args[0]
        ip = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = th.nn.MSELoss()(recons, ip)

        # TODO check the dimensionalilty
        kld_loss = th.mean(
            -0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "recon_loss": recons_loss.detach(),
            "kl_loss": -kld_loss.detach(),
        }
