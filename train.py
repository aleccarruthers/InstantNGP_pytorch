import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from mlp import MLP
from hash_map_ngp import HashMapNGP
from utilities.ngp_dataset import NGPDataset
from ngp import NGP
from utilities.metrics import psnr_comp
from utilities.data import *
from configparser import ConfigParser
from utilities.constants import Constants



def setup_dataset(config):
    """
    Creates pytorch dataset for InstantNGP implementation

    Args:
        config (dict): 
            Config file containing training parameters

    Returns:
        NGPDataset (torch.Dataset):
            Custom pytorch dataset
    """
    image_path = config[Constants.TRAIN][Constants.IMAGE_PATH]
    image = plt.imread(image_path)
    ngp_dataset = NGPDataset(image)
    img_shape = image.shape[0:2]
    return ngp_dataset, img_shape

def setup_hash_map(config, img_shape, device):
    """
    Creates HashMap pytorch module to use for image fitting

    Args:
        config (dict): 
            Config file containing training parameters
        img_shape (tuple):
            A 2-tuple decribing height, width of input image
        device (torch.device): 
            Device being used for training

    Returns:
        HashMapNGP:
            Hash map of embeddings associated with different coordinates in input image
    """
    hash_size = config[Constants.TRAIN][Constants.HASH_SIZE]
    rows, cols = img_shape
    hash_encoder = HashMapNGP(rows, hash_size_exp=hash_size, device=device)
    return hash_encoder

def setup_mlp(config, hash_encoder, output_size=3):
    """
    Setups mlp network for InstantNGP

    Args:
        config (dict):
            Config file containing training parameters
        hash_encoder (HashMapNGP): 
            Custom Pytorch module representing hash map in InstantNGP
            implementation
        output_size (int, optional): 
            Output channels. Defaults to 3.

    Returns:
        _type_: _description_
    """
    mlp = MLP(
        input_dim=hash_encoder.embedding_size,
        output_size=output_size,
        hidden_size=int(config[Constants.TRAIN][Constants.MLP_NEURONS_PER_LAYER]),
        num_hidden_layers=int(config[Constants.TRAIN][Constants.MLP_HIDDEN_LAYERS]), 
        output_act='relu'
    )
    return mlp

def setup_optimizer(config, hashmap_params, mlp_params):
    """
    Creates optimizer

    Args:
        config (dict): 
            Config file containing training parameters
        hashmap_params (list): 
            List of parameters belonging to HashMapNGP instance
        mlp_params (list): 
            List of parameters belonging to MLP instance

    Returns:
        torch.optim:
            Optimizer to be used during training
    """
    betas = (
        float(config[Constants.TRAIN][Constants.ADAM_B1]),
        float(config[Constants.TRAIN][Constants.ADAM_B2])
    )
    lr = float(config[Constants.TRAIN][Constants.LR])
    optimizer = torch.optim.Adam(
        [
            {'params': hashmap_params, 'eps':1e-15, 'betas': betas, 'lr': lr},
            {'params': mlp_params, 'weight_decay': 1e-6, 'betas': betas, 'lr': lr}
        ]
    )
    return optimizer

def eval_model(model, batches, save_dir, img_shape):
    """
    Uses the trained model to generate the gigapixel image it was trained to fit.

    Args:
        model (torch.nn.Module): 
            Trained InstantNGP Model
        batches (torch.tensor): 
            Batches of training data that make up the image
        save_dir (str): 
            Output directory for fitted image
        img_shape (tuple): 
            Height and width of training image
    """
    with torch.no_grad():
        tensor_list = []
        for bb in batches:
            d, _ = bb
            f_predictions = model(d)
            tensor_list.append(f_predictions)
        tt = torch.cat(tensor_list, axis=0)
        result = tt.T
        result*=255
        result = torch.clamp(result, 0, 255).type(torch.uint8)
        result_image = torch.reshape(result, (3, img_shape[0], img_shape[1]))
        result_image.size()
        result_image = result_image.swapaxes(0, 1)
        result_image = result_image.swapaxes(1, 2)
        result = result_image.cpu().detach()
        result_np = result.numpy()
        im_path = save_dir / f'eval_image.png'
        plt.imsave(im_path, result_np)

def main(config):
    save_dir, model_dir = setup_output_dir(config)
    ngp_dataset, img_shape = setup_dataset(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)
    hash_encoder = setup_hash_map(config, img_shape, device)
    mlp = setup_mlp(config, hash_encoder)
    hashmap_params = list(hash_encoder.parameters())
    mlp_params = list(mlp.parameters())
    model = NGP(hash_encoder=hash_encoder, mlp=mlp)
    optimizer = setup_optimizer(config, hashmap_params, mlp_params)
    hash_encoder = hash_encoder.to(device)
    mlp = mlp.to(device)
    model.to(device)
    criterion = torch.nn.MSELoss(reduction='mean')
    training_batches = get_batches(ngp_dataset, batch_size=config[Constants.TRAIN][Constants.BATCH_SIZE])

    best_loss = torch.inf
    for epoch in range(int(config[Constants.TRAIN][Constants.EPOCHS])):
        epoch_loss = 0
        epoch_psnr = 0
        for i, data_batch in enumerate(training_batches):
            coords, pixel_vals = data_batch
            optimizer.zero_grad()
            #with torch.cuda.amp.autocast():
            predictions = model(coords)
            loss = criterion(predictions, pixel_vals)
            loss.backward()
            optimizer.step()
            psnr_val = psnr_comp(predictions, pixel_vals)
            epoch_loss += loss.item()
            epoch_psnr += psnr_val.item()
        batches = i + 1
        if epoch_loss/batches < best_loss:
            saved_models = list(model_dir.glob(f'model*'))
            if len(saved_models)>1:
                Path.unlink(saved_models[0])
            model_path = model_dir / f'model_loss_{epoch_loss/batches}.pt'
            torch.save(model.state_dict(), model_path)
            best_loss = epoch_loss/batches
        print(f'Iteration: {epoch}, Loss: {epoch_loss/batches}, PSNR: {epoch_psnr/batches}')
    eval_model(model, training_batches, save_dir, img_shape)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)
    main(config)
