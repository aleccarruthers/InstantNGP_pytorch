import torch
from torch.utils.data import Dataset


class NGPDataset(Dataset):
    """
    Creates a torch Dataset for a single image.
    Used in conjunction with torch.utils.data.DataLoader
    to get x,y pairs for training
    """
    def __init__(self, image, device="cuda"):
        self.device = device
        if not isinstance(image, torch.Tensor):
            self.image = torch.Tensor(image)
        self.n_rows = self.image.shape[0]
        self.n_cols = self.image.shape[1]

        self.x = self.coords()
        self.y = self.colors()
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        """
        Returns a single example from the Dataset.
        Args:
            index (int): The index of the example to return. 
        """
        return self.x[index], self.y[index]

    def __len__(self):
        """
        Returns the length of the Dataset.
        """
        return self.len
      
    def coords(self):
        """
        Creates a tensor for the pixel locations of an image.
        """
        rows = torch.arange(0, self.n_rows)
        cols = torch.arange(0, self.n_cols)
        row_coords, col_coords = torch.meshgrid((rows, cols), indexing='ij')
        row_coords_flat = row_coords.flatten()
        col_coords_flat = col_coords.flatten()
        row_coords_flat = row_coords_flat[:, None]
        col_coords_flat = col_coords_flat[:, None]
        coord_tensor = torch.concat((row_coords_flat, col_coords_flat), dim=1)
        return coord_tensor.to(self.device).type(torch.float32)
    
    def colors(self):
        """
        Creates a tensor for the (R,G,B) color channels of an image.
        """
        normalized_px_vals = self.image / torch.max(self.image)
        color_vals = normalized_px_vals[self.x.type(torch.LongTensor)[:,0],self.x.type(torch.LongTensor)[:,1],:]
        return color_vals.to(self.device)
    