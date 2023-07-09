import torch

def psnr_comp(predictions, target, max_val = 1):
    """
    Calculates the PSNR between the predictions and the source image

    Args:
        predictions (torch.tensor): 
            Output pixel value predictions from the model
        target (torch.tensor):
            Known pixel values for each coordinate in the source image
        max_val (int, optional): 
            Max pixel value in source image. Defaults to 1.

    Returns:
        torch.tensor:
            PSNR of predictions based on the input target
    """
    max_val2 = max_val**2
    mse = torch.nn.functional.mse_loss(predictions, target)
    psnr = 10 * torch.log10(max_val2/mse)
    return psnr