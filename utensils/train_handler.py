from __future__ import print_function
import argparse
import subprocess
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pandas as pd
from .others import calc_cost
import datetime


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = []
    train_rmse = []
    for batch_idx, data in enumerate(train_loader):
        control_point, control_changed_point, cloud, relation, label, label2 = [i.to(device) for i in data]

        optimizer.zero_grad()
        output1, output2 = model(control_point, control_changed_point, cloud, relation)
        loss1 = F.l1_loss(output1, label)
        loss2 = F.l1_loss(output2, label2)
        loss = loss1 + loss2
        loss3 = torch.sqrt(F.mse_loss(output1, label))
        loss4 = torch.sqrt(F.mse_loss(output2, label2))
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        train_rmse.append((loss3 + loss4).detach().cpu().numpy())
    return np.mean(train_loss), np.mean(train_rmse)


def test(model, device, test_loader):
    model.eval()
    test_loss = []
    test_rmse = []
    with torch.no_grad():
        for data in test_loader:
            control_point, control_changed_point, cloud, relation, label, label2 = [i.to(device) for i in data]
            output1, output2 = model(control_point, control_changed_point, cloud, relation)
            loss1 = F.l1_loss(output1, label)
            loss2 = F.l1_loss(output2, label2)
            loss = loss1 + loss2
            loss3 = torch.sqrt(F.mse_loss(output1, label))
            loss4 = torch.sqrt(F.mse_loss(output2, label2))
            test_loss.append(loss.item())
            test_rmse.append((loss3 + loss4).detach().cpu().numpy())
        test_loss = np.mean(test_loss)
        return test_loss, np.mean(test_rmse)


def init_args(batch_size, epochs, lr, gpu_idx):
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=lr, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=9, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if use_cuda:
        if gpu_idx is None:
            gpu_idx = choice_gpu()
        elif gpu_idx >= torch.cuda.device_count():
            gpu_idx = 0
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 6,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    return args, device, train_kwargs, test_kwargs


def choice_gpu():
    gpu_utilization = []
    for i in range(torch.cuda.device_count()):
        cmd = f"nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits --id={i}"
        result = subprocess.check_output(cmd, shell=True)
        gpu_utilization.append(int(result))
    return int(np.argmin(gpu_utilization))


def init_dataset(base_dir, dataset_folder, train_percent, train_kwargs, test_kwargs):
    control_dataset = torch.load(base_dir + dataset_folder + '/control.pt')
    control_changed_dataset = torch.load(base_dir + dataset_folder + '/control_changed.pt')
    cloud_dataset = torch.load(base_dir + dataset_folder + '/cloud.pt')
    relation_dataset = torch.load(base_dir + dataset_folder + '/relation.pt')
    cloud_changed_dataset = torch.load(base_dir + dataset_folder + '/cloud_changed.pt')
    control_completion_dataset = torch.load(base_dir + dataset_folder + '/control_completion.pt')
    n_train = int(len(control_dataset) * train_percent)
    train_loader = DataLoader(TensorDataset(control_dataset[:n_train], control_changed_dataset[:n_train],
                                            cloud_dataset[:n_train], relation_dataset[:n_train],
                                            cloud_changed_dataset[:n_train], control_completion_dataset[:n_train]),
                              **train_kwargs)
    test_loader = DataLoader(TensorDataset(control_dataset[n_train:], control_changed_dataset[n_train:],
                                           cloud_dataset[n_train:], relation_dataset[n_train:],
                                           cloud_changed_dataset[n_train:], control_completion_dataset[n_train:]),
                             **test_kwargs)
    return train_loader, test_loader


def save_result(model, dataset_folder, train_history):
    # time stamp
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    # save plot result
    save_folder = time_stamp + '_' + dataset_folder
    os.makedirs('result/' + save_folder)
    df = pd.DataFrame(train_history)
    df.columns = ['train_loss', 'test_loss', 'train_rmse', 'test_rmse']
    df.to_csv(f'result/{save_folder}/plot_result_epochs({df.shape[0]}).csv', index=False)
    # save model
    torch.save(model.state_dict(), f"result/{save_folder}/{model.__class__.__name__}.pt")


@calc_cost
def run(Module, dataset_folder, base_dir='data/generate_dataset/',
        train_percent=0.8, batch_size=8, epochs=200, lr=1e-3, gpu_idx=None):
    args, device, train_kwargs, test_kwargs = init_args(batch_size, epochs, lr, gpu_idx)
    train_loader, test_loader = init_dataset(base_dir, dataset_folder, train_percent, train_kwargs, test_kwargs)
    model = Module(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
    train_history = np.zeros([args.epochs, 4])
    for epoch in range(args.epochs):
        train_loss, train_rmse = train(model, device, train_loader, optimizer)
        test_loss, test_rmse = test(model, device, test_loader)
        train_history[epoch, 0] = train_loss
        train_history[epoch, 1] = test_loss
        train_history[epoch, 2] = train_rmse
        train_history[epoch, 3] = test_rmse
        print(f"epoch: {epoch + 1}\n"
              f"train loss: {train_loss}\ttrain rmse: {train_rmse}\n"
              f"test loss: {test_loss}\ttest rmse: {test_rmse}")
        scheduler.step()
    save_result(model, dataset_folder, train_history)
