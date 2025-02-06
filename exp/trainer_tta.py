from copyreg import constructor
from utils.compute import cosine_similarity
from data_provider.data_factory import data_provider
from exp.trainer_basic import Exp_Basic
from utils.tools import cal_MF1score
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
import torch.nn.functional as F

warnings.filterwarnings('ignore')
 
class exp_tta(Exp_Basic):
    def __init__(self, args):
        super(exp_tta, self).__init__(args)

    def _build_model(self):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        # model init
        model = self.model_dict[self.args.model].AdaWarp(self.args).float()
        self.model2 = self.model_dict[self.args.model].AdaWarp(self.args).float()
        self.classes=self.args.num_classes
        self.V=self.args.V
        self.args_stage2=copy.deepcopy(self.args)          
        self.args_stage2.stage='stage2'
        self.args_stage2.is_training=1        

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(args = self.args, flag = flag)
        return data_set, data_loader

    def _select_optimizer(self):
        param_groups = [
            {'params': self.model.ae.parameters(), 'lr': self.args.learning_rate},
            {'params': [self.model.V], 'lr': 0.07}
        ]
        model_optim = optim.Adam(param_groups)
        return model_optim

    def _select_criterion(self):
        criterion=nn.MSELoss()
        return criterion

    def vali(self, setting):
        pass
    
    def train(self, setting):
        pass

    def test(self, setting, test=1):
        #  print(
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting.replace('tta', 'stage3'), 'checkpoint.pth'), map_location='cuda:0')
            self.model.load_state_dict(checkpoint)

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        print('delta,N:', self.args.delta, self.args.N)
        preds, trues = [], []
        iter_count = 0
        time_now = time.time()
        for i, (batch_x, label) in enumerate(test_loader):
            iter_count += 1
            with torch.no_grad():
                S=1
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                delta, n = self.args.delta, self.args.N
                if self.args.tta == 0:
                    _, _, outputs = self.model(batch_x, S)
                else:
                    V_s = S - n * delta * S
                    V_pre, V_post = V_s, 0
                    logits_list, w_list = [], []
                    _, _, pre_outputs = self.model(batch_x, V_s)
                    for k in range(1, 2 * n + 1):
                        V_post = V_pre + k * delta * S
                        _, _, post_outputs = self.model(batch_x, V_post)
                        W_v = cosine_similarity(pre_outputs, post_outputs)

                        # probs = F.softmax(post_outputs, dim=1)
                        # W_v = -torch.sum(probs * torch.log(probs), dim=1).unsqueeze(1)

                        logits_list.append(post_outputs)
                        w_list.append(W_v)
                        V_pre, pre_outputs = V_post, post_outputs
                    
                    
                    logits_stacked, w_stacked = torch.stack(logits_list), torch.stack(w_list)
                    
                    w_softmax = F.softmax(w_stacked, dim=1)
                    # print(w_softmax)
                    # w_softmax.fill_(1/(2 * n))
                    w_expanded = w_softmax.expand_as(logits_stacked)
                    logits_weighted = logits_stacked * w_expanded
                    outputs = logits_weighted.sum(dim=0)

                preds.append(outputs.detach())
                trues.append(label)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        print('test shape:', preds.shape, trues.shape)
        probs = torch.nn.functional.softmax(preds)  
        print(probs.shape)
        predictions = torch.argmax(probs, dim=1).cpu()
        trues = trues.flatten().cpu()
        total_time = time.time() - time_now
        accuracy = cal_MF1score(predictions, trues,self.classes, False)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('The result of this test for MF1 is:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        
        if self.args.tta == 1:
            f.write('After tta setting as delta:{0}, N:{1}\n'.format(delta, n))
        f.write('The result of this test for MF1 is:{0}\n'.format(accuracy))
        f.write('The processing time for each sample is:{0:.3f}s\n'.format(total_time / trues.shape[0]))
        f.write('\n')
        f.write('\n')
        f.close()
        return