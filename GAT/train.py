import os 
import time 
import random 
import argparse
import glob

import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.autograd import Variable

from utils import accuracy, load_data
from model import GCN, GAT, SpGCN, SpGAT
 


if __name__  == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer'], help='Dataset to train.')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:0")

    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

    adj, features, labels = adj.to(device), features.to(device), labels.to(device)
    
    # model = GCN(nfeat = features.shape[1], nhid = args.hidden, nclass = labels.max().item()+1, dropout = args.dropout)
    # model = GAT(nfeat = features.shape[1], nhid = args.hidden,nclass = labels.max().item()+1, dropout = args.dropout, alpha = args.alpha ,nheads = args.n_heads)
    model = SpGCN(nfeat = features.shape[1], nhid = args.hidden, nclass = labels.max().item()+1, dropout = args.dropout)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = args.lr, 
                           weight_decay= args.weight_decay)
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    
    train_loss_arr =[]
    valid_loss_arr = []
    valid_acc_arr = []
    for ep in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        train_loss = criterion(output[idx_train], labels[idx_train])
        train_loss.backward()
        optimizer.step()
        train_loss_arr.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            valid_loss = criterion(output[idx_val], labels[idx_val])
            valid_loss_arr.append(valid_loss.item())
            val_acc = accuracy(output[idx_val], labels[idx_val])
        valid_acc_arr.append(val_acc.item())

        print(f"{ep}: train:{train_loss.item():.4f}, valid:{valid_loss.item():.4f}, acc:{val_acc.item():.4f}")
    
    model = model.cpu()
    torch.save({
        "train_loss_arr": train_loss_arr,
        "valid_loss_arr": valid_loss_arr,
        "valid_acc_arr": valid_acc_arr,
        "model": model.state_dict() 
    }, "save.pth")
    model.to(device)
    model.eval()
    output= model(features, adj)
    test_loss = criterion(output[idx_test], labels[idx_test])
    test_acc  = accuracy(output[idx_test], labels[idx_test])
    print(f"loss: {test_loss:.4f}, acc: {test_acc:.4f}")