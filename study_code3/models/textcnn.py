# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TextCNN, self).__init__()
        self.opt = opt
        # nn.Embedding()#随机初始化的，需要训练
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        # conv1 : [input_channel, output_channel, filter_height), padding=1]
        self.conv1 = nn.Conv1d(opt.embed_dim, opt.hidden_dim, 4, padding=1)
        #+一层attention
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices = inputs[0]
        text = self.embed(text_indices)#[batch_size, seq_len, embedding_dim]
        x = text.transpose(1,2)#[batch_size, embedding_dim, seq_len]
        x = F.relu(self.conv1(x))##[batch_size, hidden_dim, seq_len]
        x = torch.avg_pool1d(x, kernel_size=x.shape[-1]).squeeze(-1)#[batch_size, hidden_dim]
        output = self.fc(x)
        return output
