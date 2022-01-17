import torch as th
from typing import List
import pdb

from carl.context_encoders.context_encoder import ContextEncoder

class ContextAE(ContextEncoder):
    def __init__(
            self, 
            input_dim: int = 5,
            latent_dim: int = 1, 
            hidden_dims: List = [3],):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        

        self._build_network()

        # Registering  
        self.representations = None
        self.encoder.register_forward_hook(self._representation_hook)
        self.double()
    
        
    def _build_network(self) -> None:
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

    def forward(self, x) -> th.Tensor:
        
        # Extract representation from the input

        # Re-create the input
        return [self.decode(self.encode(x)), x]

    def encode(self, x) -> th.Tensor:
        """
        Pass the tensor through the encoder network
        """
        return self.encoder(x)

    def decode(self, x) -> th.Tensor:
        """
        Pass the tensor through the decoder network
        """
        return self.decoder(x)
    
    def _representation_hook(self, inst, inp, out):
        """
        Return a hook that returns the representation of the layer.
        """
        self.representations = out
    
    def get_representation(self) -> th.Tensor:
        """
        Get the recorded latent representations of the encoder
        """
        return self.representations

    def get_encoder(self) -> th.nn.Module:
        """
        Get the encoder module
        """
        return self.encoder
    
    def get_decoder(self)-> th.nn.Module:
        """
        Get the decoder module
        """
        return self.decoder

    def loss_function(  self,
                        *args, 
                        **kwargs) -> dict:
        """
        Calculate the loss the loss
        """
        recons = args[0]
        ip     = args[1]

        loss = th.nn.functional.mse_loss(recons, ip)

        return {'loss': loss}
            
