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
        self.self_attn = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp')
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices = inputs[0]
        text_len = torch.sum(text_indices != 0, dim=-1)
        x = self.embed(text_indices)
        _, (h_n, _) = self.lstm(x, text_len)
        x, _ = self.self_attn(x, x)  # 使用自注意力[batch_size, seq_len, hidden_dim]
        x = torch.div(torch.sum(x, dim=1), text_len.unsqueeze(1).float())
        out = self.fc(x)
        return out
