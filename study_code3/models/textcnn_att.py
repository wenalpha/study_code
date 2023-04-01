# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layers.attention import Attention

class TextCNN_Attn(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TextCNN_Attn, self).__init__()
        self.opt = opt
        # nn.Embedding()#随机初始化的，需要训练
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        # conv1 : [input_channel, output_channel, filter_height), padding=1]
        self.conv1 = nn.Conv1d(opt.embed_dim, opt.hidden_dim, 4, padding=1)
        #添加一层自注意力层
        self.self_attn = Attention(opt.hidden_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp')
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices = inputs[0]
        text_len =torch.sum(text_indices != 0, dim = -1)
        text = self.embed(text_indices)#[batch_size, seq_len, embedding_dim]
        x = text.transpose(1,2)#[batch_size, embedding_dim, seq_len]
        x = torch.relu(self.conv1(x))##[batch_size, hidden_dim, seq_len]
        x = x.transpose(1,2)#[batch_size, seq_len, hidden_dim]
        x, _ = self.self_attn(x, x)#使用自注意力[batch_size, seq_len, hidden_dim]
        x = torch.div(torch.sum(x, dim=1), text_len.unsqueeze(1).float())
        output = self.fc(x)
        return output
