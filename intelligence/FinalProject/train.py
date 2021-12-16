import argparse

import torch
from torch.utils.data import DataLoader

from model import ModelClass
from utils import RecommendationDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2021 AI Final Project')
    parser.add_argument('--save-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./data', help='dataset directory')
    parser.add_argument('--batch-size', default=16, help='train loader batch size')

    args = parser.parse_args()

    # instantiate model
    model = ModelClass()

    # load dataset in train folder
    train_data = RecommendationDataset(f"{args.dataset}/ratings.csv", train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    """
    Implement code for training the recommendation model
    """

    torch.save(model.state_dict(), args.save_model)
