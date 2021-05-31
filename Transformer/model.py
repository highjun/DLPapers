import torch
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np 

def positionalEncoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    
    for i in range(d_model//2):
        pe[:,2*i] = torch.sin(torch.Tensor(range(seq_len))/(10**(8*i/d_model)))
        pe[:,2*i+1] = torch.cos(torch.Tensor(range(seq_len))/(10**(8*i/d_model)))
    return pe

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_model, self.config.d_model)
        self.W_K = nn.Linear(self.config.d_model, self.config.d_model)
        self.W_V = nn.Linear(self.config.d_model, self.config.d_model)
        self.W_O = nn.Linear(self.config.d_model, self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self, query, keyvalue, mask):
        #query: bxp1xd_model, keyvalue: bxp2xd_model: torch.FloatTensor 
        #mask: bxp1xp2 : torch.BoolTensor
        b, p1, _ = query.size()
        b, p2, _ = keyvalue.size()

        q = self.W_Q(query).view(b, p1, self.config.n_head, self.config.d_k).transpose(1,2) # bxhxp1xd_k
        k = self.W_K(keyvalue).view(b, p2, self.config.n_head, self.config.d_k).transpose(1,2) # bxhxp2xd_k
        v = self.W_V(keyvalue).view(b, p2, self.config.n_head, self.config.d_k).transpose(1,2) # bxhxp2xd_k

        mask = mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1) # bxhxp1xp2

        attention = torch.matmul(q, k.transpose(-1,-2)).mul_(self.config.d_k ** (-0.5) )
        attention.masked_fill_(mask, -1e9)
        attention = F.softmax(attention, dim = -1) # bxhxp1xp2
        assert attention.shape == (b, self.config.n_head, p1, p2)
        attention_dropped = self.dropout(attention)

        output = torch.matmul(attention_dropped, v) #bxhxp1xd_k
        output = output.transpose(1,2).reshape(b, p1, -1)  #bxp1xd_model
        output = self.W_O(output) # bxp1xd_model
        
        return output, attention

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
  
        self.multihead_attn = MultiHeadAttention(config)      
        self.feed_forward = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_ff),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_ff, self.config.d_model))
        
        self.attn_norm = nn.LayerNorm(self.config.d_model)
        self.ffn_norm = nn.LayerNorm(self.config.d_model)
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.ffn_dropout = nn.Dropout(self.config.dropout)
    def forward(self, src, mask):
        # src: bxpxd_model FloatTensor
        # mask: bxpxp BoolTensor
        b, p, check_dim = src.size()
        assert check_dim == self.config.d_model
        assert mask.size() == (b, p, p)

        output, attention = self.multihead_attn(src, src, mask)
        output = self.attn_norm(src + self.attn_dropout(output))

        output_ = self.feed_forward(output)
        output = self.ffn_norm(output + self.ffn_dropout(output_))

        return output, attention
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
  
        self.self_attn = MultiHeadAttention(config)     
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_ff),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_ff, self.config.d_model))
        
        self.self_attn_norm = nn.LayerNorm(self.config.d_model)
        self.cross_attn_norm = nn.LayerNorm(self.config.d_model)
        self.ffn_norm = nn.LayerNorm(self.config.d_model)
        self.self_attn_dropout = nn.Dropout(self.config.dropout)
        self.cross_attn_dropout = nn.Dropout(self.config.dropout)
        self.ffn_dropout = nn.Dropout(self.config.dropout)
    def forward(self, src, target, self_mask, cross_mask):
        # src: bxp1xd_model FloatTensor
        # target: bxp2xd_model FloatTensor
        # self_mask: bxp2xp2 BoolTensor
        # cross_mask: bxp2xp1 BoolTensor
        b, p1, check_dim = src.size()
        p2 = target.size()[1]
        assert check_dim == self.config.d_model
        assert target.size() == (b, p2, self.config.d_model)
        assert self_mask.size() == (b, p2, p2)
        assert cross_mask.size() == (b, p2, p1)

        output, self_attention = self.self_attn(target, target, self_mask)
        output = self.self_attn_norm(target + self.self_attn_dropout(output))

        output_, cross_attention = self.cross_attn(output, src, cross_mask)
        output = self.cross_attn_norm(output + self.cross_attn_dropout(output_))

        output_ = self.feed_forward(output)
        output = self.ffn_norm(output + self.ffn_dropout(output_))

        return output, self_attention, cross_attention

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_embedding = nn.Embedding(self.config.src_vocab_size, self.config.d_model)
        self.dec_embedding = nn.Embedding(self.config.dec_vocab_size, self.config.d_model)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_enc_layer)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_dec_layer)])

        self.enc_norm = nn.LayerNorm(self.config.d_model)
        self.dec_norm = nn.LayerNorm(self.config.d_model)

        self.prediction_head = nn.Linear(self.config.d_model, self.config.dec_vocab_size)
        
    def forward(self, src, target):
        # src: b x p1 IntTensor
        # target: b x p2 IntTensor
        b, p1 = src.size()
        _, p2 = target.size()
        
        src_pad_mask = (src == self.config.pad_idx).unsqueeze(1).repeat(1, p1, 1).to(self.config.device)
        
        cross_pad_mask = (src == self.config.pad_idx).unsqueeze(1).repeat(1, p2, 1).to(self.config.device)
        
        target_pad_mask = (target == self.config.pad_idx).unsqueeze(1).repeat(1, p2, 1).to(self.config.device)
        target_subseq_mask = (torch.tril(torch.ones(p2, p2)).unsqueeze(0).repeat(b, 1, 1) == 0).to(self.config.device)
        target_mask = torch.logical_or(target_pad_mask, target_subseq_mask)


        src_emb = self.enc_embedding(src) + positionalEncoding(p1, self.config.d_model).to(self.config.device)
        target_emb = self.dec_embedding(target) + positionalEncoding(p2, self.config.d_model).to(self.config.device)

        src_self_attention_layers = []
        trg_self_attention_layers = []
        cross_attention_layers = []

        for layer in self.encoder_layers:
            src_emb, src_self_attention = layer(src_emb, src_pad_mask)
            src_self_attention_layers.append(src_self_attention)
        for layer in self.decoder_layers:
            target_emb, trg_self_attention, cross_attention = layer(src_emb, target_emb, target_mask, cross_pad_mask)
            trg_self_attention_layers.append(trg_self_attention)
            cross_attention_layers.append(cross_attention)
        output = self.prediction_head(target_emb)
        return output, src_self_attention_layers, trg_self_attention_layers, cross_attention_layers
    def criterion(self, pred, target):
        loss_func = nn.CrossEntropyLoss(ignore_index= self.config.pad_idx)
        return loss_func(pred, target)
    def inference(self, src, src_vocab, trg_vocab, device):
        src_idx = torch.tensor([src_vocab.stoi[token] for token in src]).view(1, -1)
        self.eval()
        pred_token = ["<sos>"]
        with torch.no_grad():
            for idx in range(30):
                pred_idx = torch.tensor([trg_vocab.stoi[token] for token in pred_token]).view(1, -1).to(device) 
                output = self.forward(src_idx, pred_idx)[0]
                output = output.argmax(dim= -1)#bxp
                if output[0,-1] == trg_vocab.stoi['<eos>']:
                    break
                pred_token.append(trg_vocab.itos[output[0,-1]])
        print(f"src: {src}\ntrg: {pred_token}")
    def calc_bleu(self, test_iter, trg_vocab, device):
        candidate_corpus = []
        ref_corpus = []
        self.eval()
        with torch.no_grad():
            for _, (src, trg) in enumerate(test_iter):
                src = src.to(device)
                pred_idx = torch.tensor([trg_vocab.stoi['<sos>']]).repeat(src.shape[0]).view(-1, 1)
                for idx in range(30):
                    output = model(src, pred_idx)[0]
                    output = output.argmax(dim= -1)#bxp
                    if torch.logical_or(output[:, -1] == trg_vocab.stoi['<eos>'], output[:,-1] == trg_vocab.stoi['<pad>']):
                        break
                    pred_idx = torch.cat([pred, output[:,-1:]], axis = -1)
                for i in range(pred.shape[0]):
                    pred_mask = torch.logical_not(torch.logical_or(pred_idx == trg_vocab.stoi['<eos>'], pred_idx == trg_vocab.stoi['<pad>'])).numpy()
                    pred_idx = pred_idx.cpu().detach().numpy()[pred_mask]
                    trg_mask = torch.logical_not(torch.logical_or(trg == trg_vocab.stoi['<eos>'], trg == trg_vocab.stoi['<pad>'])).numpy()
                    trg = trg.cpu().detach().numpy()[trg_mask]
                    candidate_corpus.append([trg_vocab.itos[idx] for idx in pred_idx[i][1:]])
                    ref_corpus.append([[trg_vocab.itos[idx] for idx in trg[i][1:]]])
        return bleu_score(candidate_corpus, ref_corpus)