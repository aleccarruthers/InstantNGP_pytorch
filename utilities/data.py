import torch
from pathlib import Path
from utilities.constants import Constants

def get_batches(data, batch_size):
    """
    Splits the data into approximately equal sized chunks.
    A torch data loader was not used as the images could be fit 
    entirely into GPU memory

    Args:
        data (torch.dataset):
            A torch dataset with the x attribute representing the pixel
            coordinates and y attribute being the multi channel pixel vaues
        batch_size (int): 
            Number of data points per batch

    Returns:
        list:
            List of tensor batches to use during training
    """
    coord_batches = list(torch.split(data.x, batch_size))
    pixel_val_batches = list(torch.split(data.y, batch_size))
    training_batches = list(zip(coord_batches, pixel_val_batches))
    return training_batches

def setup_output_dir(config, subfolder='model'):
    """
    Creates directories to house training results

    Args:
        config (dict): 
            Config file containing training parameters
        subfolder (str, optional): 
            Name of subfolder within 'output_dir' that contains
            the best trained model. Defaults to 'model'.

    Returns:
        tuple: 
            Tuple of output_dir path and filepath to directory that will
            contain the best trained model.
    """
    output_dir = Path(__file__).resolve().parent.parent
    output_dir = Path(output_dir) / Path(config[Constants.TRAIN][Constants.OUTPUT_DIR])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / subfolder
    model_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, model_dir