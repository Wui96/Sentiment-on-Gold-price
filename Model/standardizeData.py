from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from pathlib import Path
import sys

def standardizeData(X, SS = None, train = False):
    """Given a list of input features, standardizes them to bring them onto a homogenous scale
    Args:
        X ([dataframe]): [A dataframe of all the input values]
        SS ([object], optional): [A StandardScaler object that holds mean and std of a standardized dataset]. Defaults to None.
        train (bool, optional): [If False, means validation set to be loaded and SS needs to be passed to scale it]. Defaults to False.
    """
    
    _dir = Path(sys.path[0])
    
    if train:
        SS = StandardScaler()   
        new_X = SS.fit_transform(X)
        dump(SS, _dir / 'resource/std_scaler.bin', compress=True)
        return (new_X, SS)
    else:
        SS = load(_dir / 'resource/std_scaler.bin')
        new_X = SS.transform(X)
        return (new_X, None)