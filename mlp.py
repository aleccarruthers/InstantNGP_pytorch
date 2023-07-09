import torch
from feed_forward_layer import FeedForwardLayer

class MLP(torch.nn.Module):
    """
    Baseline MLP for InstantNGP
    """

    MIN_LAYERS = 2

    def __init__(self, input_dim, output_size, num_hidden_layers, hidden_size, output_act='relu'):
        """
        Initialize mlp that'll serve as a baseline

        Args:
            input_dim (int): 
                Dimensions of input tensor (should be two)
            output_size (int):
                Dimensions of the tensor that should be predicted for each grid coordindate.
                If RGB input image -> output_dim = 3
                If grayscale input image -> output_dim = 1      
            num_hidden_layers (int): 
                Number of interior linear layers. Input and output layers do not count to 
                this total.
            hidden_size (int):
                Dimensions to use for hidden layers
            output_act (str):
                String used to indicate whether the output activation
                from the mlp should be a relu or sigmoid.
        """
        super().__init__()
        assert  num_hidden_layers >= self.MIN_LAYERS
        self.initial_layer = FeedForwardLayer(input_dim, hidden_size)
        if output_act == 'relu':
            self.output_layer = FeedForwardLayer(hidden_size, output_size)
        else:
            self.output_layer = FeedForwardLayer(hidden_size, output_size, activation='sigmoid')
        self.mlp_layers = [self.initial_layer]
        for _ in range(num_hidden_layers - self.MIN_LAYERS):
            hidden_layer = FeedForwardLayer(hidden_size, hidden_size)
            self.mlp_layers.append(hidden_layer)
        self.mlp_layers.append(self.output_layer)
        self.sequential_layers = torch.nn.Sequential(*self.mlp_layers)

    def forward(self, inputs):
        """
        Forward function for baseline mlp

        Args:
            inputs (torch.Tensor): 
                Tensor of image grid coordinates. Just passing 2d coordinates 
                [0,0], [0,1], [0,2], ....

        Returns:
            torch.Tensor:
                Output tensor from model forward
        """
        x = self.sequential_layers(inputs)
        return x