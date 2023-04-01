import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TextCNN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        output_channel = opt.hidden_dim
        self.conv = nn.Sequential(
            # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
            nn.Conv2d(1, output_channel, (3, opt.embed_dim)),
            nn.ReLU(),
            # pool : ((filter_height, filter_width))
            nn.MaxPool2d((2, 1)),
        )
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
      '''
      X: [batch_size, sequence_length]
      '''
      text_indices = inputs[0]
      batch_size = text_indices.shape[0]
      text = self.embed(text_indices).unsqueeze(1)
      text = text.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
      conved = self.conv(text) # [batch_size, output_channel*1*1]
      flatten = conved.view(batch_size, -1)
      output = self.fc(flatten)
      return output


# class TextCNN(nn.Module):
#     def __init__(self, embedding_matrix, opt):
#         super(TextCNN, self).__init__()
#         self.opt = opt
#         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.conv = nn.Sequential(
#             # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
#             nn.Conv2d(1, opt.hidden_dim, (3, opt.embed_dim)),
#             nn.ReLU(),
#             # pool : ((filter_height, filter_width))
#             nn.MaxPool2d((2, 1)),
#         )
#         # fc
#         self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
#
#     def forward(self, inputs):
#       '''
#       X: [batch_size, sequence_length]
#       '''
#       text_indices = inputs[0]
#       text = self.embed(text_indices).unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
#       print(text.shape, 'bbbbbbbbbb')
#       text = self.conv(text) # [batch_size, output_channel*1*1]
#       print(text.shape, 'aaaaaaaaaaaaa')
#       flatten = text.view(text.shape[0], -1)
#       print(flatten.shape)
#       output = self.fc(flatten)
#       return output