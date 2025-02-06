from data_provider.data_loader import  FDAloader, SSCloader, HARloader
from torch.utils.data import DataLoader
import os

data_dict = {
    'FDA': FDAloader,
    'SSC': SSCloader,
    'HAR': HARloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    normalized_path = os.path.normpath(args.root_path)
    root_path = os.path.join('./datasets', normalized_path)

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'classification':
        drop_last = True
        data_set = Data(
            root_path=root_path,
            flag=flag,
            win_size=args.win_size,
            win_step=args.win_step,
            train_domain=args.train_domain,
            test_domain=args.test_domain
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader