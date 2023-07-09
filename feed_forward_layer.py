import torch

class FeedForwardLayer(torch.nn.Module):

    "Single feed forward layer"

    def __init__(self, input_dim, output_dim, initialization = torch.nn.init.kaiming_normal_, activation='relu'):
        """
        Initialize linear layer

        Args:
            input_dim (int): 
                Dimensions in the input coordinates (should always be two)
            output_dim (int): 
                Dimensions of output tensor (pixel values the model should learn per grid coordinate).
                This is also defined by whether the input image was RGB or grayscale. 
                Grayscale -> 1
                RGB -> 3
            initialization (torch.nn.init, optional): 
                Initialization for linear layers to provide more stable training.
                Defaults to torch.nn.init.kaiming_normal_.
            activation (str):
                Activation function to use after fully connected layer. Only supported activations are
                'relu' and 'sigmoid'.
        """
        super().__init__()
        self.dense_layer = torch.nn.Linear(input_dim, output_dim)
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.Sigmoid()
        initialization(self.dense_layer.weight)

    def forward(self, input):
        """
        Forward for dense layer

        Args:
            input (torch.Tensor)

        Returns:
            torch.Tensor 
        """
        x = self.activation(self.dense_layer(input))
        return x
    