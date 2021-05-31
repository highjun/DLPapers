from torch.optim import Adam
import torch

import os
from time import time
from tqdm import tqdm
import argparse

from dataloader import getDatas
from model import SAT
import utils

def main(args):
    if not os.path.exists("./exps"):
        os.mkdir('./exps')
    device = torch.device(f"cuda:{args.device_id}")
    train_loader, valid_loader, field = getDatas(args.dataset, batch_size = args.batch_size)
    args.dict_size = len(field.vocab)
    torch.save(field, f"exps/{args.n_exp}_field.pth")
    model = SAT(args)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr= args.lr)

    train_loss_arr, valid_loss_arr = [], []
    best_val_loss = float('inf')
    n_iter = 0
    for ep in range(args.ep):
        if args.debug and ep ==1:
            break
        start_ep = time()
        train_loss = 0.
        pbar = tqdm(enumerate(train_loader), total= len(train_loader))
        model.train()
        ep_train_loss_arr = []
        for idx, batch in pbar:
            if args.debug and idx ==2:
                break
            optimizer.zero_grad()
            img, caption = batch.img, batch.cap 
            img, caption = img.to(device), caption.to(device)
            pred = model(img, caption[:,:-1])[0]
            loss = model.criterion(pred, caption[:,1:])
            loss.backward()
            optimizer.step()
            n_iter += 1
            train_loss = loss.item()
            pbar.set_postfix({'train_loss':train_loss})
            ep_train_loss_arr.append(train_loss)
        train_loss_arr += ep_train_loss_arr
        model.eval()
        valid_loss =0.
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_loader),total = len(valid_loader)):
                if args.debug and idx ==2:
                    break
                img, caption = batch.img, batch.cap
                img, caption = img.to(device), caption.to(device)
                pred = model(img, caption[:,:-1])[0]
                loss = model.criterion(pred, caption[:,1:])
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        valid_loss_arr.append(valid_loss)
        train_loss = sum(ep_train_loss_arr)/len(ep_train_loss_arr)
        if best_val_loss> valid_loss:
            best_val_loss = valid_loss
            utils.save(model,f"./exps/{args.n_exp}_best.pth")
        utils.save(model, f"./exps/{args.n_exp}_recent.pth")
        torch.save({
            "train_loss_arr": train_loss_arr,
            "valid_loss_arr": valid_loss_arr,
            "ep": ep,
            "n_iter": n_iter,
        },f"./exps/{args.n_exp}_log.pth")
        print(f"[{ep},{n_iter}]: {utils.timeCheck(start_ep)}s, train_loss ={train_loss:.2f}, valid_loss ={valid_loss:.2f}")


                

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Show, Attend, and Tell")

    parser.add_argument('--dataset', type=str, default="flickr8k")
    parser.add_argument('--n_worker',type = int,default = 4)

    parser.add_argument('--batch_size',type=int, default=2)
    parser.add_argument('--lr',type= float,default=1e-4)
    parser.add_argument('--ep',type= int,default = 100)

    parser.add_argument('--dict_size', type = int, default = None)
    parser.add_argument('--dim_embed', type = int, default = 128)
    parser.add_argument('--dim_lstm_hid', type= int, default = 256)
    parser.add_argument('--n_conv_channel', type= int, default=512)
    parser.add_argument('--feature_map_size', type= int, default= 196)

    parser.add_argument('--debug',type= bool,default = False)
    parser.add_argument('--device_id',type = int,default = 0)
    

    args = parser.parse_args()
    args.attn_mlp = [args.dim_lstm_hid + args.n_conv_channel, 128 ,1]
    args.init_ch_mlp = [args.n_conv_channel, 2*args.dim_lstm_hid]
    # args.init_ch_mlp = [args.n_conv_channel, 256 , 2*args.dim_lstm_hid]
    if not args.debug:
        with open("./exp_setting.txt", "a+") as f:
            f.seek(0)
            n_exp = len(f.readlines())
            args.n_exp = n_exp
            f.write(str(args)+"\n")   
    main(args)
