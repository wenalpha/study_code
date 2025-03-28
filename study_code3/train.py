# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import DatesetReader
from models import TextCNN, LSTM, TextCNN_Attn, LSTM_Attn

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        dataset = DatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)

        self.train_data_loader = BucketIterator(data=dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.valid_data_loader = BucketIterator(data=dataset.valid_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=dataset.test_data, batch_size=opt.batch_size, shuffle=False)

        self.model = opt.model_class(dataset.embedding_matrix, opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    #中间过程使用验证集进行验证，得出最佳模型并保存
                    val_acc, val_f1 = self._evaluate_acc_f1(self.valid_data_loader)
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                    if val_f1 > max_val_f1:
                        increase_flag = True
                        max_val_f1 = val_f1
                        if self.opt.save and val_f1 > self.global_f1:
                            self.global_f1 = val_f1
                            torch.save(self.model.state_dict(), 'state_dict/'+self.opt.model_name+'_'+self.opt.dataset+'.pkl')
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, val_acc, val_f1))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
            print('max_val_acc：', max_val_acc, '\tmax_val_f1：', max_val_f1)
        #使用测试集合测试模型效果
        test_acc, test_f1 = self._evaluate_acc_f1(self.test_data_loader)
        return test_acc, test_f1

    def _evaluate_acc_f1(self, data_loader):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        labels = [0, 1, 2, 3, 4]#根据不同分类来修改
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=labels, average='macro')
        return test_acc, f1

    def run(self, repeats=3):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')

        test_acc_avg = 0
        test_f1_avg = 0#recall , precisionf1 = 2 * pr * r / (p +ｒ)
        for i in range(repeats):
            print('repeat: ', (i+1))
            f_out.write('repeat: '+str(i+1))
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            test_acc, test_f1 = self._train(criterion, optimizer)
            print('test_acc: {0}     test_f1: {1}'.format(test_acc, test_f1))
            f_out.write('test_acc: {0}, test_f1: {1}'.format(test_acc, test_f1))
            test_acc_avg += test_acc
            test_f1_avg += test_f1
            print('#' * 100)
        print("max_test_acc_avg:", test_acc_avg / repeats)
        print("max_test_f1_avg:", test_f1_avg / repeats)

        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lstm_attn', type=str, help='textcnn, lstm, textcnn_attn, lstm_attn')
    parser.add_argument('--dataset', default='sst', type=str, help='sst, cola')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.003, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=50, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--polarities_dim', default=5, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args()

    model_classes = {
        'textcnn': TextCNN,
        'lstm': LSTM,
        'textcnn_attn': TextCNN_Attn,
        'lstm_attn': LSTM_Attn,
    }
    input_colses = {
        'textcnn': ['text_indices'],
        'lstm': ['text_indices'],
        'textcnn_attn': ['text_indices'],
        'lstm_attn': ['text_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()
