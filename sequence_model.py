# -*- coding: utf-8 -*-
"""
@author: Rashid Haffadi
"""

import torch
from torch import nn
from torch import autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def load_model(output_size, num_of_ngrams=200000,pretrained=False, path="F:/workspace/PFE/Models/SequenceModel/", name="checkpoint1.state_dict"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceModel(num_of_ngrams=num_of_ngrams, output_size=output_size, 
                      hidden_size=32, 
                      embedding_dim=16).to(device).eval()
    
    if pretrained:
        state_dict = torch.load(path + name, map_location=device)
        model.load_state_dict(state_dict)
        
        model.chars_embedding = nn.Embedding(num_embeddings=num_of_ngrams, padding_idx=0, embedding_dim=embedding_dim)
    return model

# {"seqtolang":}
def from_config(config:dict):
        model = SequenceModel(num_of_ngrams=config["num_of_ngrams"], output_size=config["output_size"], 
                    hidden_size=config["hidden_size"], embedding_dim=config["embedding_dim"], drp=config["drp"], 
                    n_rounds=config["n_rounds"], n_layers=config["n_layers"], bidirectional=config["bidirectional"], 
                    batch_first=config["batch_first"])
        return model

class SequenceModel(nn.Module):
    ""
    def __init__(self, num_of_ngrams=159694, output_size=21, 
                 hidden_size=64, embedding_dim=64, drp=0.2, 
                 n_rounds=2, n_layers=2, bidirectional=True, 
                 batch_first=True):
        ""
        super().__init__()
        self.chars_embedding = nn.Embedding(num_embeddings=num_of_ngrams, padding_idx=0, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(num_layers=n_layers, input_size=embedding_dim, 
                            hidden_size=hidden_size, bidirectional=bidirectional, 
                            batch_first=batch_first)
        if bidirectional: dout = din = hidden_size*2
        else: din = dout = hidden_size
        self.layers = [self.round(din, dout, drp) for k in range(n_rounds)]
        self.linear = nn.Linear(dout, output_size)

        # self.linear1 = nn.Linear(hidden_size*2, output_size*4)
        # self.bn1 = nn.BatchNorm1d(output_size*4)
        # self.dropout1 = nn.Dropout(drp)

        # self.linear2 = nn.Linear(output_size*4, output_size*2)
        # self.bn2 = nn.BatchNorm1d(output_size*2)
        # self.dropout2 = nn.Dropout(drp)

        # self.linear3 = nn.Linear(output_size*2, output_size)
        self.with_gpu()

    def round(self, din, dout, drp):
        return nn.Sequential(nn.Linear(din, dout), 
                       nn.BatchNorm1d(dout),
                       nn.Dropout(drp),
                       nn.ReLU())

    def with_gpu(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def smouthing(self, o):
        a = o.argmax(1)
        a = F.one_hot(a, num_classes=o.shape[1]).float()
        return a.clone().detach().requires_grad_(True)

    def forward(self, x):
        out = self.chars_embedding(x)

        # print("" + str(out.shape))
        out, hidden = self.lstm(out)
        out = out.sum(1)/x.shape[1]

        for l in self.layers:
            out = l(out)

        # # round 1
        # out = self.linear1(out)
        # out = self.bn1(out)
        # out = self.dropout1(out)
        # out = F.relu(out)

        # # round 2
        # out = self.linear2(out)
        # out = self.bn2(out)
        # out = self.dropout2(out)
        # our = F.relu(out)

        out = self.linear(out)
        # out = self.linear3(out)
        out = F.softmax(out, dim=1)
        return out
    
if __name__ == "__main__":
    model = from_config(dict(num_of_ngrams=159694, output_size=21, 
                                      hidden_size=64, embedding_dim=64, drp=0.2, 
                                      n_rounds=2, n_layers=2, bidirectional=True, 
                                      batch_first=True))
    print(model)
