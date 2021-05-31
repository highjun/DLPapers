import numpy as np 
import torch
import torch.nn as nn
from time import time

from torchtext.data.metrics import bleu_score

class Args(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, device, ignore_idx, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.ignore_idx = ignore_idx
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pad = target == self.ignore_idx
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def time_check(start):
    total_time = round(time() - start)
    min, seconds = divmod(total_time, 60)
    print("{:02}:{:02}".format(int(min),int(seconds)))


def attention_visualization(model, iteration, enc_self_head, dec_self_head, cross_head):
    iteration.batch_size = 1
    sample_idx = np.random.randint(0,high = len(iteration.data()),size= 1)[0]
    data_iter = iter(iteration)
    batch = None
    for i in range(sample_idx):
        batch = next(data_iter)
    src, trg = batch.src, batch.trg
    src_token, trg_token = [SRC.vocab.itos[idx] for idx in src[0]], [TRG.vocab.itos[idx] for idx in trg[0]]
    model.eval()
    pred = None
    with torch.no_grad():
        output, enc_self_attn, dec_self_attn, cross_attn = model(src, trg)
        pred = output.argmax(dim = -1).squeeze()[:-1] #p
        pred = [TRG.vocab.itos[idx] for idx in pred]
        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (18,6))
        fig.suptitle("Attention Visualization")
        im = ax1.imshow(enc_self_attn[-1][0, enc_self_head][1:-1,1:-1].cpu().detach().numpy())
        ax1.set_xticks(range(len(src_token[1:-1])))
        ax1.set_xticklabels(src_token[1:-1], fontsize = 14)
        ax1.set_yticks(range(len(src_token[1:-1])))
        ax1.set_yticklabels(src_token[1:-1], fontsize = 14)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
        ax1.set_title("Encoder")
        
        im = ax2.imshow(dec_self_attn[-1][0, dec_self_head][1:-1,1:-1].cpu().detach().numpy())
        ax2.set_xticks(range(len(trg_token[1:-1])))
        ax2.set_xticklabels(trg_token[1:-1], fontsize = 14)
        ax2.set_yticks(range(len(trg_token[1:-1])))
        ax2.set_yticklabels(trg_token[1:-1], fontsize = 14)
        ax2.set_title("Decoder")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

        im = ax3.imshow(cross_attn[-1][0, cross_head][1:-1,1:-1].cpu().detach().numpy())
        ax3.set_yticks(range(len(trg_token[1:-1])))
        ax3.set_yticklabels(trg_token[1:-1], fontsize = 14)
        ax3.set_xticks(range(len(src_token[1:-1])))
        ax3.set_xticklabels(src_token[1:-1], fontsize = 14)
        ax3.set_title("Cross")
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    print(f"src: {src_token}\ntrg: {trg_token}\npred_as_train: {pred}")
    iteration.batch_size = config.batch_size