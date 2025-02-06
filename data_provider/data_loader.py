import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

class Normalizer(object):
    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        if self.norm_type == "standardization":
            if self.mean is None:
                # self.mean = df.mean()
                # self.std = df.std()
                self.mean = torch.mean(df,dim=(0,2))
                self.std = torch.std(df,dim=(0,2))
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


class FDAloader(Dataset):
    def __init__(self, root_path, train_domain,test_domain,win_step=512, win_size=512,flag=None):
        super().__init__()
        self.flag = flag
        self.step = win_step
        self.win_size = win_size
        self.max_seq_len = win_size
        train_dataset = torch.load(os.path.join(root_path, train_domain))
        test_dataset = torch.load(os.path.join(root_path, test_domain))     
        # Load samples
        train_data = train_dataset["samples"]
        test_data = test_dataset["samples"]
        # Load labels
        train_labels = train_dataset.get("labels")
        test_labels = test_dataset.get("labels")
        # Slice into windows
        train_data = self.preprocess(train_data)
        test_data = self.preprocess(test_data)
        # Convert to torch tensor
        if train_labels is not None and isinstance(train_labels, np.ndarray):
            train_labels = torch.from_numpy(train_labels)
        if test_labels is not None and isinstance(test_labels, np.ndarray):
            test_labels = torch.from_numpy(test_labels)
        if isinstance(train_data, np.ndarray):
            train_data = torch.from_numpy(train_data)
        if isinstance(test_data, np.ndarray):
            test_data = torch.from_numpy(test_data)    
        
        self.train_data = train_data.float()
        self.train_labels = train_labels.long() if train_labels is not None else None
        self.test_data = test_data.float()
        self.test_labels = test_labels.long() if test_labels is not None else None

    def preprocess(self, data):
        # change the window size dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        win_size = self.win_size
        step = self.step
        num_windows = (data.shape[1] - win_size) // step + 1
        stacked_data = []
        # slice the instance into pieces
        for i in range(num_windows):
            start = i * step
            end = start + win_size
            data_slice = data[:,start:end]
            stacked_data.append(data_slice)
        data = torch.stack(stacked_data, dim=1)
        # Normalize data
        normalizer = Normalizer()
        
        data = normalizer.normalize(data)
        return data

    def __getitem__(self, index):
        if self.flag == "train":
            x = self.train_data[index]
            y = self.train_labels[index] if self.train_labels is not None else None
            return x, y

        elif (self.flag == 'test'):
            x = self.test_data[index]
            y = self.test_labels[index] if self.test_labels is not None else None
            return x, y
        else:
            x = self.test_data[index]
            y = self.test_labels[index] if self.test_labels is not None else None
            return x, y


    def __len__(self):
        if self.flag == "train":
            return self.train_data.shape[0]
        elif (self.flag == 'test'):
            return self.test_data.shape[0]
        else:
            return self.test_data.shape[0]
        
class SSCloader(Dataset):
    def __init__(self, root_path, train_domain,test_domain,win_step=512, win_size=512,flag=None):
        super().__init__()
        self.flag = flag
        self.step = win_step
        self.win_size = win_size
        self.max_seq_len = win_size
        train_dataset = torch.load(os.path.join(root_path, train_domain))
        test_dataset = torch.load(os.path.join(root_path, test_domain))     
        # Load samples
        train_data = train_dataset["samples"]
        test_data = test_dataset["samples"]
        # Load labels
        train_labels = train_dataset.get("labels")
        test_labels = test_dataset.get("labels")
        # Slice into windows

        # Convert to torch tensor
        if train_labels is not None and isinstance(train_labels, np.ndarray):
            train_labels = torch.from_numpy(train_labels)
        if test_labels is not None and isinstance(test_labels, np.ndarray):
            test_labels = torch.from_numpy(test_labels)

        if isinstance(train_data, np.ndarray):
            train_data = torch.from_numpy(train_data)
            if train_data.shape[-1] == 1:
                train_data = train_data.squeeze(-1)
            train_data = F.pad(train_data, (36, 36), mode='constant', value=0)
        if isinstance(test_data, np.ndarray):
            test_data = torch.from_numpy(test_data)
            if test_data.shape[-1] == 1:
                test_data = test_data.squeeze(-1) 
            test_data = F.pad(test_data, (36, 36), mode='constant', value=0)
        
        train_data = self.preprocess(train_data)
        test_data = self.preprocess(test_data)
        
        self.train_data = train_data.float()
        self.train_labels = train_labels.long() if train_labels is not None else None
        self.test_data = test_data.float()
        self.test_labels = test_labels.long() if test_labels is not None else None

    def preprocess(self, data):
        # change the window size dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        win_size = self.win_size
        step = self.step
        num_windows = (data.shape[1] - win_size) // step + 1
        stacked_data = []
        # slice the instance into pieces
        for i in range(num_windows):
            start = i * step
            end = start + win_size
            data_slice = data[:,start:end]
            stacked_data.append(data_slice)
        data = torch.stack(stacked_data, dim=1)
        # Normalize data
        normalizer = Normalizer()
        
        data = normalizer.normalize(data)
        return data

    def __getitem__(self, index):
        if self.flag == "train":
            x = self.train_data[index]
            y = self.train_labels[index] if self.train_labels is not None else None
            return x, y

        elif (self.flag == 'test'):
            x = self.test_data[index]
            y = self.test_labels[index] if self.test_labels is not None else None
            return x, y
        else:
            x = self.test_data[index]
            y = self.test_labels[index] if self.test_labels is not None else None
            return x, y


    def __len__(self):
        if self.flag == "train":
            return self.train_data.shape[0]
        elif (self.flag == 'test'):
            return self.test_data.shape[0]
        else:
            return self.test_data.shape[0]

class HARloader(Dataset):
    def __init__(self, root_path, train_domain, test_domain,win_step=512, win_size=512,flag=None):
        super().__init__()
        self.flag = flag
        self.step = win_step
        self.win_size = win_size
        self.max_seq_len = win_size
        train_dataset = torch.load(os.path.join(root_path, train_domain))
        test_dataset = torch.load(os.path.join(root_path, test_domain))     
        # Load samples
        train_data = train_dataset["samples"]
        test_data = test_dataset["samples"]
        # Load labels
        train_labels = train_dataset.get("labels")
        test_labels = test_dataset.get("labels")
    
        if train_labels is not None and isinstance(train_labels, np.ndarray):
            train_labels = torch.from_numpy(train_labels)
        if test_labels is not None and isinstance(test_labels, np.ndarray):
            test_labels = torch.from_numpy(test_labels)
        if isinstance(train_data, np.ndarray):
            train_data = torch.from_numpy(train_data)
        if isinstance(test_data, np.ndarray):
            test_data = torch.from_numpy(test_data)  
        
        data_mean_train= torch.mean(train_data, dim=(0, 2))
        data_std_train = torch.std(train_data, dim=(0, 2))
        self.transform_train = transforms.Normalize(mean=data_mean_train, std=data_std_train)

        data_mean_test= torch.mean(test_data, dim=(0, 2))
        data_std_test = torch.std(test_data, dim=(0, 2))
        self.transform_test = transforms.Normalize(mean=data_mean_test, std=data_std_test)

        # Slice into windows
        train_data = self.preprocess(train_data)
        test_data = self.preprocess(test_data)
          
        self.train_data = train_data.float()
        self.train_labels = train_labels.long() if train_labels is not None else None
        self.test_data = test_data.float()
        self.test_labels = test_labels.long() if test_labels is not None else None

    def preprocess(self, data):
        # change the window size dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        win_size = self.win_size
        step = self.step
        num_windows = (data.shape[2] - win_size) // step + 1
        stacked_data = []
        # slice the instance into piecesd
        for i in range(num_windows):
            start = i * step
            end = start + win_size
            data_slice = data[:,:,start:end]
            stacked_data.append(data_slice)
        data = torch.stack(stacked_data, dim=1)
        data=data.squeeze(1)
        return data

    def __getitem__(self, index):
        if self.flag == "train":
            x = self.train_data[index]
            x = self.transform_train(self.train_data[index].reshape(9, -1, 1)).reshape(self.train_data[index].shape)
            y = self.train_labels[index] if self.train_labels is not None else None
            return x, y

        elif (self.flag == 'test'):
            x = self.test_data[index]
            x = self.transform_test(self.test_data[index].reshape(9, -1, 1)).reshape(self.test_data[index].shape)
            y = self.test_labels[index] if self.test_labels is not None else None
            return x, y
        else:
            x = self.test_data[index]
            x = self.transform_test(self.test_data[index].reshape(9, -1, 1)).reshape(self.test_data[index].shape)
            y = self.test_labels[index] if self.test_labels is not None else None
            return x, y


    def __len__(self):
        if self.flag == "train":
            return self.train_data.shape[0]
        elif (self.flag == 'test'):
            return self.test_data.shape[0]
        else:
            return self.test_data.shape[0]
        