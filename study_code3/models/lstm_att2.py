# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
from layers.attention import Attention


class LSTM_Attn(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM_Attn, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        #+attention
        self.last_attn = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp')
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices = inputs[0]
        text_len = torch.sum(text_indices != 0, dim=-1)
        x = self.embed(text_indices)
        x_len = torch.sum(text_indices != 0, dim=-1)
        out, (h_n, ct) = self.lstm(x, x_len)
        ct = ct.squeeze(0)#相当于是最后一个神经元的输出[batch_size, hidden_dim]
        x = h_n[0]#最后一层的特征
        x, _ = self.last_attn(x, ct)  # 使用将最后一个神经元的特征作为q注意力[batch_size, hidden_dim]
        x = x.squeeze(1)
        out = self.fc(x)
        return out
