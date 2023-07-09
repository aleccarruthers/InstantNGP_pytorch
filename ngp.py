import torch

class NGP(torch.nn.Module):
    def __init__(self, hash_encoder, mlp):
        """
        A Pytorch implementation of the Instant NGP CUDA model by
        NVIDIA. Unlike the original implementation, this neural representation
        network is solely capable of fitting 2D images.

        Args:
            hash_encoder (torch.nn.Module):
                A multi resolution hash encoder used to create the mlp input embeddings
                for each coordinate in the source image.
            mlp (torch.nn.Module): 
                A sequential mlp pytorch model. To stick with the goal of the paper,
                this should be a relatively light weight mlp.
        """
        super().__init__()
        self.hash_encoder = hash_encoder
        self.mlp = mlp

    def forward(self, tensor):
        """
        Runs forward for model training

        Args:
            tensor (torch.tensor):
                Tensor of input coordinates from the source image (num_pixels x 2)

        Returns:
            tensor (torch.tensor): 
                Tensor of equal length to the input with three columns. Each column represents the 
                predicted color channel value at the specific coordinate.
        """
        x = self.hash_encoder(tensor)
        x = self.mlp(x)
        return x