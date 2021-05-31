from torch.optim import Adam
import torch

import os
from time import time
from tqdm import tqdm
import argparse


from model import Transformer
from dataloader import getDataLoader
import utils

def main(args):
    if not os.path.exists("./exps"):
        os.mkdir('./exps')
    device = torch.device(f"cuda:{args.device_id}")
    train_loader, valid_loader, test_loader, src_vocab, trg_vocab = getDataLoader(args.data_type, args.batch_size)
    args.dict_size = len(trg_vocab)
    args.pad_idx = src_vocab['<pad>']
    torch.save(src_vocab, f"exps/src_vocab.pth")
    torch.save(trg_vocab, f"exps/trg_vocab.pth")
    model = Transformer(args)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr= args.lr, betas =args.betas)

    train_loss_arr, valid_loss_arr, test_bleu_arr = [], [], []
    best_val_loss = float('inf')
    n_iter = 0
    for ep in range(args.ep):
        if args.debug and ep ==1:
            break
        start_ep = time()
        pbar = tqdm(enumerate(train_loader), total= len(train_loader))
        model.train()
        ep_train_loss_arr = []
        for idx, (src, trg) in pbar:
            if args.debug and idx ==2:
                break
            optimizer.zero_grad()
            src, trg = src.to(device), trg.to(device)
            pred = model(src, trg[:,:-1])[0]
            loss = model.criterion(pred, trg[:,1:])
            loss.backward()
            optimizer.step()
            n_iter += 1
            pbar.set_postfix({'train_loss':loss.item()})
            ep_train_loss_arr.append(loss.item())
        train_loss_arr += ep_train_loss_arr
        model.eval()
        valid_loss =0.
        with torch.no_grad():
            for idx, (src, trg) in tqdm(enumerate(valid_loader),total = len(valid_loader)):
                if args.debug and idx ==2:
                    break
                src, trg = src.to(device), trg.to(device)
                pred = model(src, trg[:,:-1])[0]
                loss = model.criterion(pred, trg[:,1:])
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        valid_loss_arr.append(valid_loss)
        test_bleu = model.calc_bleu(test_loader, trg_vocab, device)
        test_bleu_arr.append(test_bleu)
        train_loss = sum(ep_train_loss_arr)/len(ep_train_loss_arr)
        if best_val_loss> valid_loss:
            best_val_loss = valid_loss
            utils.save(model,f"./exps/{args.n_exp}_best.pth")
        utils.save(model, f"./exps/{args.n_exp}_recent.pth")
        torch.save({
            "train_loss_arr": train_loss_arr,
            "valid_loss_arr": valid_loss_arr,
            "test_bleu_arr": test_bleu_arr
            "ep": ep,
            "n_iter": n_iter,
        },f"./exps/{args.n_exp}_log.pth")
        print(f"[{ep},{n_iter}]: {utils.timeCheck(start_ep)}s, train_loss ={train_loss:.2f}, valid_loss ={valid_loss:.2f}, test_bleu: {test_bleu:.4f}")


                

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Attention is All you Need!")

    parser.add_argument('--dataset', type=str, default="Multi30k")
    parser.add_argument('--n_worker',type = int,default = 4)

    parser.add_argument('--batch_size',type=int, default=2)
    parser.add_argument('--lr',type= float,default=1e-5)
    parser.add_argument('--betas', type = tuple, default = (0.9, 0.999))
    parser.add_argument('--epoch',type= int, default = 100)
    parser.add_argument('--warm_up', type = int, default=1000)
    parser.add_argument('--smoothing', type = float,default = 0.1)

    parser.add_argument('--d_k', type = int, default = 64)
    parser.add_argument('--n_head', type= int, default = 8)
    parser.add_argument('--dropout', type= float, default=0.1)
    parser.add_argument('--n_enc_layer', type=int, default=6)
    parser.add_argument('--n_dec_layer', type=int, default=6)
    parser.add_argument('--d_ff', type= int, default= 2048)

    parser.add_argument('--debug',type= bool,default = False)
    parser.add_argument('--device_id',type = int,default = 0)
    

    args = parser.parse_args()
    args.d_model = args.d_k * args.n_head
    if not args.debug:
        with open("./exp_setting.txt", "a+") as f:
            f.seek(0)
            n_exp = len(f.readlines())
            args.n_exp = n_exp
            f.write(str(args)+"\n")  
    main(args)
