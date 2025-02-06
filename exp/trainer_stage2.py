from copyreg import constructor
from data_provider.data_factory import data_provider
from exp.trainer_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_MF1score
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
warnings.filterwarnings('ignore')


class exp_stage2(Exp_Basic):
    def __init__(self, args):
        super(exp_stage2, self).__init__(args)

    def _build_model(self):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        model = self.model_dict[self.args.model].AdaWarp(self.args).float()
        self.classes=self.args.num_classes
        self.V=self.args.V
        self.args_stage1=copy.deepcopy(self.args)          
        self.args_stage1.stage='stage1'
        self.args_stage1.is_training=1                    

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(args = self.args, flag = flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.backboneNetwork.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)#[B,1]
                y_ae,y_unet,outputs = self.model(batch_x,self.V)
                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)
                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds) 
        predictions = torch.argmax(probs, dim=1).cpu() 
        trues = trues.flatten().cpu()
        accuracy = cal_MF1score(predictions, trues,self.classes)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='test')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        load_set_stage1 = '{}_{}_{}_{}_{}_{}'.format(
        self.args.model_id,
        self.args.model,
        self.args.data,
        self.args_stage1.stage,
        self.args.des, 0)
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + load_set_stage1, 'checkpoint.pth')))

        print('loading pretrained reconstructor parameters...')
        self.model.to(self.device)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                y_ae,y_unet,outputs = self.model(batch_x,self.V)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.backboneNetwork.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} "
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model



    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
            
        if test:
            print('loading model')
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        preds = []
        trues = []
        iter_count = 0
        for i, (batch_x, label) in enumerate(test_loader):
            iter_count += 1
            with torch.no_grad():
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                y_ae,y_unet,outputs = self.model(batch_x,self.V)
                preds.append(outputs.detach())
                trues.append(label)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)


        print('test shape:', preds.shape, trues.shape)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu()
        trues = trues.cpu()
        accuracy = cal_MF1score(predictions, trues,self.classes,True)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('MF1score:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('The MF1score of this test is:{0}\n'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
