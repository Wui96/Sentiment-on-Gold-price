import torch

def get_preds(generator, model):
    """Given a pytorch neural network model and a generator object, extracts predictions and returns the same

    Args:
        generator ([object]): [A pytorch dataloader which holds inputs on which we wanna predict]
        model ([object]): [A pytorch model with which we will predict stock prices on input data]

    """
    all_preds = []
    all_labels = []
    all_ips = []
    for xb, yb in generator:
        ips = xb.unsqueeze(0)
        ops = model.predict(ips)
        all_preds.append(ops)
        all_ips.append(ips)
        all_labels.append(yb)
    return (torch.cat(all_preds), torch.cat(all_labels))