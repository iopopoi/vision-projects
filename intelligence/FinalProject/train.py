import argparse
import pickle
from numpy.core.arrayprint import format_float_positional
import torch
from torch.nn.modules.sparse import Embedding
from torch.utils.data import DataLoader

from torch import nn

import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import ModelClass
from model import EmbeddingLayers

from utils import RecommendationDataset

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':

    # hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epoch = 50
    Reguli = 0

    # --  setting -----
    #  1. model type - "Embedding or origin Matrix factorization"
    type = "Embedding"
    # type = "origin"

    #  2. validation data size
    size_valdata = batch_size * 500

    parser = argparse.ArgumentParser(description='2021 AI Final Project')
    parser.add_argument('--save-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./data', help='dataset directory')
    parser.add_argument('--batch-size', default=batch_size, help='train loader batch size') # batch size 설정

    args = parser.parse_args()

    # instantiate model
    load = False

    if load == False:
        if type == "origin" :
            model = ModelClass()
        elif type == "Embedding" :
            model = EmbeddingLayers()
            
    else:
        with open('data.pickle', 'rb') as f:
            model = pickle.load(f)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=Reguli) # weight_decay(regularization)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)            

    # load dataset in train folder
    # train_data = RecommendationDataset(f"{args.dataset}/ratings.csv", train=True)
    data = RecommendationDataset(f"{args.dataset}/ratings.csv", train=True)
    train, val = torch.utils.data.random_split(data, [data.len - size_valdata, size_valdata])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)

    # data size
    n_users, n_items, n_ratings = data.get_datasize()

    """
    Implement code for training the recommendation model
    """

    # data for chart
    costL_train = []
    costL_val = []

    for epoch in range(num_epoch):
        cost = 0
        for users, items, ratings in train_loader:
            optimizer.zero_grad()
            ratings_pred = model(users, items)
            loss = criterion(ratings_pred, ratings)
            loss.backward()
            optimizer.step()
            cost += loss.item() * len(ratings)

        cost /= (data.len - size_valdata)
        costL_train.append(cost)
        print(f"Epoch: {epoch}")
        print("train cost: {:.6f}".format(cost))

        cost_val = 0
        for users, items, ratings in val_loader:
            ratings_pred = model(users, items)
            loss = criterion(ratings_pred, ratings)
            cost_val += loss.item() * len(ratings)
        
        cost_val /= size_valdata
        costL_val.append(cost_val)
        print("val cost: {:.6f}".format(cost_val))

        # early stopping
        if epoch > 0 :
            if cost_val > costL_val[epoch-1] :
                print("validation cost increase")
                print("---[ early stoping]---------")
                print("Last val cost:",cost_val)
                break

    with open('data.pickle', 'wb') as f:
        pickle.dump(model, f)
    
    torch.save(model.state_dict(), args.save_model)

    x_train = [i for i in range(len(costL_train))]
    x_val = [i for i in range(len(costL_val))]
    plt.plot(x_train, costL_train, x_val, costL_val)
    plt.legend(["Train","Val"])
    print("Minimum train cost:",min(costL_train))
    print("Minimum val cost:",min(costL_val))
    plt.show()
