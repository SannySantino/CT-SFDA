import argparse
import os
import torch
from exp.trainer_stage1 import exp_stage1
from exp.trainer_stage2 import exp_stage2
from exp.trainer_stage3 import exp_stage3
from exp.trainer_tta import exp_tta
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='classification',help='task name')
    parser.add_argument('--is_training', type=int,  default=0, help='status')
    parser.add_argument('--model_id', type=str,  default='FDA_512_0', help='model id')
    parser.add_argument('--model', type=str, default='OOD2',help='model name')

    # data loader
    parser.add_argument('--data', type=str,  default='FDA', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./FD/', help='root path of the data file')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    #AdaWarp config
    parser.add_argument('--train_domain', type=str, default='train_0.pt', help=' ')
    parser.add_argument('--test_domain', type=str, default='train_2.pt', help=' ') 
    parser.add_argument('--win_size', type=int, default=512, help=' ')
    parser.add_argument('--win_step', type=int, default=512, help=' ') 
    parser.add_argument('--input_channels', type=int, default=10, help=' ')
    parser.add_argument('--kernel_size', type=int, default=3, help=' ')
    parser.add_argument('--stride', type=int, default=1, help=' ')
    parser.add_argument('--num_classes', type=int, default=3, help=' ')
    parser.add_argument('--mid_channels', type=int, default=64, help=' ')
    parser.add_argument('--final_out_channels', type=int, default=128, help=' ')
    parser.add_argument('--features_len', type=int, default=16, help=' ')
    parser.add_argument('--padding_length', type=int, default=1000, help=' ')
    parser.add_argument('--stage', type=str, default='stage1', help=' ')
    parser.add_argument('--V', type=float, default=1, help=' ')
    parser.add_argument('--tta', type=float, default=0, help=' ')
    parser.add_argument('--delta', type=float, default=0.0, help=' ')
    parser.add_argument('--N', type=int, default=1, help=' ')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)
    if args.stage == 'stage1':
        Exp = exp_stage1
    elif args.stage == 'stage2':
        Exp = exp_stage2
    elif args.stage == 'stage3':
        Exp = exp_stage3
    elif args.stage == 'tta':
        Exp = exp_tta

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.stage,
                args.des, ii)

            print('||start training : {}||'.format(setting))
            exp.train(setting)

            # print('||testing : {}||'.format(setting))
            # exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.stage,
                args.des, ii)

        exp = Exp(args)  # set experiments
        print('||testing : {}||'.format(setting))
        exp.test(setting,test=1)


        torch.cuda.empty_cache()
