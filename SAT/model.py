import torch.nn as nn
import torch
import numpy as np 
from torchvision.models import vgg19_bn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as T
import utils

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.225,0.225,0.225]),
])

class AttentionLSTMCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_embed = config.dim_embed
        self.dim_lstm_hid = config.dim_lstm_hid
        self.dim_ctxt = config.n_conv_channel
        self.T = nn.Linear(self.dim_embed + self.dim_lstm_hid + self.dim_ctxt, 4*self.dim_lstm_hid)
    def forward(self, embedding, hidden, ctxt, cell):
        '''
        embedding: Bxm
        hidden: Bxn
        cell: Bxn
        ctxt: BXD
        '''
        _cal = self.T(torch.cat([embedding, hidden, ctxt], dim = -1))
        ifo,g = torch.split(_cal, 3*self.dim_lstm_hid , dim = -1)
        ifo, g = F.sigmoid(ifo), F.tanh(g)
        i,f,o = torch.split(ifo, self.dim_lstm_hid, dim = -1)
        nxt_cell = f * cell + i * g
        nxt_hidden = o * F.tanh(nxt_cell)

        return nxt_cell, nxt_hidden
    def debug(self):
        n_batch = 2
        embed = torch.randn(n_batch, self.dim_embed)
        hid = torch.randn(n_batch, self.dim_lstm_hid)
        cell = torch.randn(n_batch, self.dim_lstm_hid)
        ctxt = torch.randn(n_batch, self.dim_ctxt)
        output = self.forward(embed, hid, ctxt, cell)
        assert output[0].shape == cell.shape and output[1].shape == hid.shape 

class MLP(nn.Module):
    def __init__(self, dims:list):
        super().__init__()
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        layers = []
        for idx in range(len(dims)-1):
            layers.append(nn.Linear(dims[idx], dims[idx+1]))
            if idx != len(dims)-1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
    def debug(self):
        n_batch = 10
        _input = torch.randn(n_batch, self.input_dim)
        output = self.layers(_input)
        assert output.shape ==(n_batch, self.output_dim)
class Attn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_lstm_hid = config.dim_lstm_hid
        self.feature_map_size = config.feature_map_size
        self.n_conv_channel = config.n_conv_channel
        self.attn_mlp = MLP(config.attn_mlp)
    def forward(self, features, hidden):
        '''
        features: BxD(C)xL(FeatureMap)
        hidden: Bxh
        '''
        hidden = hidden.unsqueeze(1).repeat(1, features.shape[-1],1)#BxLxh
        features = features.transpose(1,2)#BxLxD
        attn_map = self.attn_mlp(torch.cat([features, hidden], dim = -1))#BxLx1
        attn_map = F.softmax(attn_map, dim = 1) #BxLx1
        ctxt = (attn_map * features).sum(dim = 1)#BxD
        attn_map = attn_map.squeeze(-1)
        return ctxt, attn_map
    def debug(self):
        n_batch = 2 
        hidden = torch.randn(n_batch, self.dim_lstm_hid)
        features = torch.randn(n_batch,self.n_conv_channel, self.feature_map_size)
        output = self.forward(features, hidden)
        assert output[0].shape == (n_batch, self.n_conv_channel) and output[1].shape == (n_batch, self.feature_map_size)

class SAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dict_size = config.dict_size
        self.dim_lstm_hid = config.dim_lstm_hid
        self.dim_embed = config.dim_embed
        self.convNet = vgg19_bn(pretrained= True).features[:-1]
        utils.freeze(self.convNet)
        self.embedding = nn.Embedding(config.dict_size, config.dim_embed)
        self.attn = Attn(config)
        self.scale_attn = nn.Sequential(
            nn.Linear(self.dim_lstm_hid, config.n_conv_channel),
            nn.Sigmoid()
        )
        self.init_ch = MLP(config.init_ch_mlp)
        self.cell = AttentionLSTMCell(config)
        self.projection = nn.Linear(config.dim_lstm_hid, config.dict_size)
    def forward(self, img, caption):
        '''
        img: Bx(C=3)xHxW
        caption: BxP(0~K-1 중 하나)
        '''
        seq_len = caption.shape[1]
        embedding = self.embedding(caption)# BxPxembed
        feature = self.convNet(img).flatten(start_dim = -2, end_dim = -1)#BXC(channel)xL
        h, c = self.init_ch(feature.mean(dim = -1)).split(self.dim_lstm_hid,dim = -1)#Bxhidden
        attn_matrices = []
        hiddens = []
        for idx in range(seq_len):
            ctxt, attn_matrix = self.attn(feature,h)
            attn_matrices.append(attn_matrix)
            ctxt *= self.scale_attn(h)
            h,c = self.cell(embedding[:,idx], h, ctxt, c)
            hiddens.append(h)
        hiddens = torch.stack(hiddens , dim = 1) 
        hiddens = self.projection(hiddens)#BxPxdict_size
        return hiddens, attn_matrices
    def debug(self):
        n_batch ,p = 2, 10
        input_seq = torch.randint(0, self.dict_size,(n_batch, p))
        img = torch.randn(n_batch, 3, 224,224)
        output = self.forward(img, input_seq)
        print(output[1][0].shape)
        assert output[0].shape == (n_batch, p, self.dict_size)
        assert output[1][0].shape == (n_batch, 196)
    def criterion(self, pred, caption):
        loss_func = nn.CrossEntropyLoss(ignore_index= 1)
        loss = loss_func(pred.transpose(1,2), caption)
        return loss
    def prediction(self, img, device,field):
        self.eval()
        with torch.no_grad():
            img = Image.open("../data/Flickr8k/Images/1002674143_1b742ab4b8.jpg")
            img = transform(img).view(1, 3, 224, 224).to(device)
            start = field.vocab.stoi["<sos>"]
            caption_tok = [start,]
            cap = torch.tensor(start).view(1,1).to(device)
            caption_str = ["<sos>",]
            for idx in range(20):
                pred, attn_matrices= self.forward(img, cap)
                pred_arg = pred.argmax(dim = -1)[0,-1]#BxP
                next_word = field.vocab.itos[pred_arg]
                caption_str.append(next_word)
                caption_tok.append(pred_arg.item())
                cap = torch.tensor(caption_tok).view(1,-1).to(device)
        return caption_str,attn_matrices