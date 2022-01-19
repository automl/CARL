import torch as th
from typing import List
from carl.context_encoders.context_encoder import ContextEncoder


class ContextAE(ContextEncoder):
    """
    Implementation of an Autoencoder that learns to reconstruct a context vector.

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
        self,
        input_dim: int = 5,
        latent_dim: int = 1,
        hidden_dims: List[int] = [3],
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self._build_network()

        # Registering
        self.representations = None
        self.double()

    def _build_network(self) -> None:
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

        modules.append(th.nn.Linear(input_dim, self.latent_dim))
        modules.append(th.nn.ReLU())

        self.encoder = th.nn.Sequential(*modules)

        # Make the decoder
        modules = []

        hidden_dims.reverse()
        input_dim = self.latent_dim

        for h_dim in hidden_dims:
            modules.append(th.nn.Linear(input_dim, h_dim))
            modules.append(th.nn.ReLU())
            input_dim = h_dim

        modules.append(th.nn.Linear(input_dim, self.input_dim))
        modules.append(th.nn.ReLU())

        self.decoder = th.nn.Sequential(*modules)

    def _representation_hook(self, inst, inp, out):
        """
        Return a hook that returns the representation of the layer.
        """
        self.representations = out

    def forward(self, x) -> List[th.Tensor]:
        """
        Takes a tensor, or a batch of tensors, passes it through the encoder,
        records a representation, and then decodes the latent representations

        Returns
        -------
            recon: th.Tensor
                Reconstructed context vector
            x: th.Tensor
                Input context vector

        """

        self.representations = self.encode(x)
        recon = self.decode(self.representations)

        return [recon, x]

    def encode(self, x) -> th.Tensor:
        """
        Pass the tensor through the encoder network

        Parameters
        ----------
        x: th.Tensor
            Input context vector
        """
        return self.encoder(x)

    def decode(self, x) -> th.Tensor:
        """
        Pass the tensor through the decoder network

        Parameters
        ----------
        x: th.Tensor
            Latent context vector
        """
        return self.decoder(x)

    def get_representation(self, context) -> th.Tensor:
        """
        Get the recorded latent representations of a passed context vector
        """

        return self.encode(context)

    def get_encoder(self) -> th.nn.Module:
        """
        Get the encoder module

        Returns
        -------
        encoder: th.nn.Module
            Encoder module
        """
        return self.encoder

    def get_decoder(self) -> th.nn.Module:
        """
        Get the decoder module

        Returns
        -------
        decoder: th.nn.Module
            Decoder module
        """
        return self.decoder

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Calculate the loss

        Parameters
        ----------
        *args: Any
            Arguments to be passed to the loss function

        **kwargs: Any
            Keyword arguments to be passed to the loss function

        Returns
        -------
        loss: dict
            Dictionary containing the loss and the loss components

        """
        recons = args[0]
        ip = args[1]

        loss = th.nn.functional.mse_loss(recons, ip)

        return {"loss": loss}
