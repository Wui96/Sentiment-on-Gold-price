import torch
from stockTickerDataset import stockTickerDataset

def getDL(x, y, params):
    """Given the inputs, labels and dataloader parameters, returns a pytorch dataloader
    Args:
        x ([list]): [inputs list]
        y ([list]): [target variable list]
        params ([dict]): [Parameters pertaining to dataloader eg. batch size]
    """
    training_set = stockTickerDataset(x, y)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    return training_generator