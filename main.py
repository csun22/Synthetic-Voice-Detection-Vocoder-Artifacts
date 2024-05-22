import argparse
import sys
import os
import numpy as np
import librosa
import yaml
import random
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from model import RawNet
from core_scripts.startup_config import set_random_seed
from pdb import set_trace
from tqdm import tqdm
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE = 24000

class Dataset_LibriSeVoc(Dataset):
    
    def __init__(self, dataset_path, split = 'train'):
            self.dataset_path = dataset_path
            
            y_list_train = []
            path_list_train = []

            y_list_test = []
            path_list_test = []

            y_list_dev = []
            path_list_dev = []


            for subset_name in os.listdir(dataset_path):
                if subset_name.startswith('gt'):
                    # print(subset_name)
                    path_list =  []
                    y_list = []
                    subset_path = os.path.join(dataset_path,subset_name)
                    # print(len(os.listdir(subset_path)), 0)
                    for file_name in os.listdir(subset_path):
                        path_list.append(os.path.join(subset_path, file_name))
                        y_list.append(0)

                    y_list_train.extend(y_list[0:7920])
                    y_list_dev.extend(y_list[7920:10560])
                    y_list_test.extend(y_list[10560:13201])

                    path_list_train.extend(path_list[0:7920])
                    path_list_dev.extend(path_list[7920:10560])
                    path_list_test.extend(path_list[10560:13201])

            i = 1
            for subset_name in os.listdir(dataset_path):
                if not subset_name.startswith('gt'):
                    # print(subset_name)
                    path_list =  []
                    y_list = []
                    subset_path = os.path.join(dataset_path,subset_name)
                    # print(len(os.listdir(subset_path)), i)
                    for file_name in os.listdir(subset_path):
                        path_list.append(os.path.join(subset_path, file_name))
                        y_list.append(i)

                    y_list_train.extend(y_list[0:7920])
                    y_list_dev.extend(y_list[7920:10560])
                    y_list_test.extend(y_list[10560:13201])

                    path_list_train.extend(path_list[0:7920])
                    path_list_dev.extend(path_list[7920:10560])
                    path_list_test.extend(path_list[10560:13201])

                    i += 1
            
            
            self.y_list_train = y_list_train
            self.path_list_train = path_list_train

            self.y_list_test = y_list_test
            self.path_list_test = path_list_test

            self.y_list_dev = y_list_dev
            self.path_list_dev = path_list_dev
            
            self.split = split
            
            print('Load data from {}'.format(self.dataset_path))

    def __len__(self):
            if self.split == 'train':
                return(len(self.path_list_train))
            if self.split == 'dev':
                return(len(self.path_list_dev))
            if self.split == 'test':
                return(len(self.path_list_test))


    def __getitem__(self, index):
            self.cut=SAMPLE_RATE*4
            if self.split == 'train':
                path = self.path_list_train[index]
                Y = self.y_list_train[index]
                X, fs = librosa.load(path, sr=None)
                X_pad = pad(X,self.cut)
                x_inp = Tensor(X_pad)
                y_inp = Y
                return x_inp, y_inp, y_inp ==  0
            if self.split == 'dev':
                path = self.path_list_dev[index]
                Y = self.y_list_dev[index]
                X, fs = librosa.load(path, sr=None)
                X_pad = pad(X,self.cut)
                x_inp = Tensor(X_pad)
                y_inp = Y
                return x_inp, y_inp, y_inp == 0
            if self.split == 'test':
                path = self.path_list_test[index]
                Y = self.y_list_test[index]
                X, fs = librosa.load(path, sr=None)
                X_pad = pad(X,self.cut)
                x_inp = Tensor(X_pad)
                y_inp = Y
                return x_inp, y_inp, y_inp == 0

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/your/path/to/LibriSeVoc/')
    parser.add_argument('--model_save_path', type=str, default='/your/path/to/models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    args = parser.parse_args()

    data_path = args.data_path
    model_save_path = args.model_save_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay

    # load dataset
    train_set = Dataset_LibriSeVoc(split = 'train', dataset_path = data_path)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)

    dev_set = Dataset_LibriSeVoc(split = 'dev', dataset_path = data_path)

    dev_dataloader = DataLoader(dev_set, batch_size=batch_size, shuffle=True, drop_last=False)

    # load model config
    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)

    # load cuda
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # init model
    model = RawNet(parser1['model'], device)
    model =(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    LAMDA = 0.5

    if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

    def evaluate_accuracy(dev_loader, model, device):
        num_correct = 0.0
        num_total = 0.0
        model.eval()
        for batch_x, batch_y_multi, batch_y_binary in dev_loader:
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            
            batch_y_binary = batch_y_binary.view(-1).type(torch.int64).to(device)
            batch_y_multi = batch_y_multi.view(-1).type(torch.int64).to(device)
            
            batch_out_binary, batch_out_multi = model(batch_x)
            
            _, batch_pred = batch_out_binary.max(dim=1)
            num_correct += (batch_pred == batch_y_binary).sum(dim=0).item()
            
        return 100 * (num_correct / num_total)


    def train_epoch(train_loader, model, lr, optim, device, lamda):
        running_loss = 0
        num_correct_binary = 0.0
        num_correct_multi = 0.0
        num_total = 0.0
        ii = 0
        model.train()

        #set objective (loss) functions
        # weight = torch.FloatTensor([0.1, 0.9]).to(device)
        # weight = torch.FloatTensor([1-lamda]+[lamda/6]*6).to(device)
        criterion_binary = nn.CrossEntropyLoss()
        criterion_multi = nn.CrossEntropyLoss()
        
        for batch_x, batch_y_multi, batch_y_binary in tqdm(train_loader,total=len(train_loader)):
            #print(batch_x.shape, batch_y_binary.shape, batch_y_multi.shape)
            batch_size = batch_x.size(0)
            num_total += batch_size
            ii += 1
            
            batch_x = batch_x.to(device)
            batch_y_binary = batch_y_binary.view(-1).type(torch.int64).to(device)
            batch_y_multi = batch_y_multi.view(-1).type(torch.int64).to(device)
            
            batch_out_binary, batch_out_multi = model(batch_x)
            #print(batch_out_binary, batch_out_multi)
            #print(batch_y_binary, batch_y_multi)
            
            batch_loss = lamda * criterion_binary(batch_out_binary, batch_y_binary) + (1- lamda) * criterion_multi(batch_out_multi, batch_y_multi)
            
            #print(batch_loss)
            
            # binary acc
            _, batch_pred_binary = batch_out_binary.max(dim=1)
            num_correct_binary += (batch_pred_binary == batch_y_binary).sum(dim=0).item()
            
            #multi acc
            _, batch_pred_multi = batch_out_multi.max(dim=1)
            num_correct_multi += (batch_pred_multi == batch_y_multi).sum(dim=0).item()
            
            if ii % 10 == 0:
                out_write = 'training multi accuracy: {:.2f}, training binary accuracy: {:.2f}'.format(
                    (num_correct_multi/num_total)*100, (num_correct_binary/num_total)*100)
                
            running_loss += (batch_loss.item() * batch_size)
            
            optim.zero_grad()
            batch_loss.backward()
            optim.step()
        
        running_loss /= num_total
        train_accuracy = ((num_correct_binary+num_correct_multi)/num_total)*50
        return running_loss, train_accuracy, out_write


best_acc = 99
for epoch in range(num_epochs):
    running_loss, train_accuracy, out_write = train_epoch(train_dataloader, model, lr, optimizer, device, lamda = LAMDA)
    valid_accuracy = evaluate_accuracy(dev_dataloader, model, device)
    print(out_write)
    print('epoch: {} -loss: {}  - valid binary accuracy: {:.2f}'.format(epoch, running_loss, valid_accuracy))
    if valid_accuracy > best_acc:
        print('best model find at epoch', epoch)
    best_acc = max(valid_accuracy, best_acc)
    torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))