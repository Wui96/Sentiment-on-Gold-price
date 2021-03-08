# Imports
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import torch.nn as nn
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.utils.data as data_utils
import math
from pathlib import Path
import sys

from scrapNews import scrapNews
from scrapFeatures import scrapFeatures
from curateData import curateData
from forecasterModel import forecasterModel
from getDL import getDL
from getPreds import get_preds
from standardizeData import standardizeData
from stockTickerDataset import stockTickerDataset
from train_val_split import train_val_split
from dbconnect import dbconnect
from getHeadline import getHeadline
from getFeatures import getFeatures
from savePred import savePred
from checkUpdate import checkUpdate

# Parameter for loading model
N = 11503
hidden_dim = 100
rnn_layers = 2
dropout = 0

# Check if predictions have been done for today
# Exit program if True
if checkUpdate():
    sys.exit()
else:
    scrapNews()
    scrapFeatures()

# Resource directory path
_dir = Path(sys.path[0]) / 'resource'    

# Preprocess headline
headlines = getHeadline()
vectorizer = pickle.load(open(_dir / "vectorizer.pickle", "rb"))
tfidf_feature = vectorizer.transform(headlines).todense().tolist()[0]

# Read other features
features = getFeatures()
ips = features + tfidf_feature
ips, _ = standardizeData(pd.DataFrame([ips]))

 _dir = Path(sys.path[0]) / 'resource' / 'model.pth'

# Load model
model2 = forecasterModel(N, hidden_dim, rnn_layers, dropout)
model2.load_state_dict(torch.load(_dir))
model2.eval()

# Get prediction and save it in database
pred = model2(torch.from_numpy(ips.astype(np.float32)).unsqueeze(0))
savePred(pred.detach().numpy()[0][0])