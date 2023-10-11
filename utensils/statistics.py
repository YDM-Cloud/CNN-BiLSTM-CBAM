import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from modules.cnn_bilstm_cbam import CNN_BiLSTM_CBAM
from modules.cnn_bilstm_sa import CNN_BiLSTM_SA
from modules.cnn_gru import CNN_GRU
from modules.cnn_gru_cbam import CNN_GRU_CBAM
from modules.cnn_lstm_cbam import CNN_LSTM_CBAM
from modules.cnn_lstm_sa import CNN_LSTM_SA
from .predict_handler import compensation


def temp_predict(control, control_changed, cloud, relation, model, std, mean,
                 cloud_changed, control_completion):
    # predict
    with torch.no_grad():
        para1 = control.unsqueeze(0)
        para2 = control_changed.unsqueeze(0)
        para3 = cloud.unsqueeze(0)
        para4 = relation.unsqueeze(0)
        predict_output1, predict_output2 = model(para1, para2, para3, para4)
    # inverse standardization
    predict_cloud_changed = (predict_output1.numpy() * std + mean)[0]
    predict_control_completion = (predict_output2.numpy() * std + mean)[0]
    train_cloud_changed = cloud_changed.numpy() * std + mean
    train_control_changed = control_changed.numpy() * std + mean
    train_control_completion = control_completion.numpy() * std + mean
    compensation_control_completion = compensation(train_control_changed, predict_control_completion)
    diff_predict_cloud_changed = np.linalg.norm(predict_cloud_changed - train_cloud_changed, axis=1)
    diff_predict_control_completion = np.linalg.norm(predict_control_completion - train_control_completion, axis=1)
    diff_compensation_control_completion = np.linalg.norm(compensation_control_completion - train_control_completion,
                                                          axis=1)
    return diff_predict_cloud_changed, diff_predict_control_completion, diff_compensation_control_completion


def statistics_compensation_result():
    cnn_bilstm_cbam_model_folder = '2023-07-22_19-17-04-131753_n_train(100000)n_control(20)n_cloud(1208)point_only(0)n_loss_control(2,0.5)standardization(1)'
    data_folder = 'n_train(100000)n_control(20)n_cloud(1208)point_only(0)n_loss_control(2,0.5)standardization(1)'
    data_folder = f'data/generate_dataset/{data_folder}/'

    cnn_bilstm_cbam_model = eval(f'CNN_BiLSTM_CBAM(device="cpu")')
    cnn_bilstm_cbam_model.load_state_dict(torch.load(f'result/{cnn_bilstm_cbam_model_folder}/CNN_BiLSTM_CBAM.pt',
                                                     map_location=torch.device('cpu')))
    cnn_bilstm_cbam_model.eval()
    # load dataset
    cnn_bilstm_cbam_control = torch.load(data_folder + 'control.pt')
    cnn_bilstm_cbam_control_changed = torch.load(data_folder + 'control_changed.pt')
    cnn_bilstm_cbam_cloud = torch.load(data_folder + 'cloud.pt')
    cnn_bilstm_cbam_relation = torch.load(data_folder + 'relation.pt')
    cnn_bilstm_cbam_cloud_changed = torch.load(data_folder + 'cloud_changed.pt')
    cnn_bilstm_cbam_control_completion = torch.load(data_folder + 'control_completion.pt')
    cnn_bilstm_cbam_mean = torch.load(data_folder + 'mean.pt').numpy()
    cnn_bilstm_cbam_std = torch.load(data_folder + 'std.pt').numpy()

    predict_diff = [[] for i in range(20)]
    compensation_diff = [[] for i in range(20)]

    for i in tqdm(range(100000)):
        check_data = cnn_bilstm_cbam_control[i] * cnn_bilstm_cbam_std + cnn_bilstm_cbam_mean
        loss_indices = torch.argwhere(torch.all(check_data == 0, dim=1)).numpy().flatten()
        if loss_indices.shape[0] == 0:
            continue

        cnn_bilstm_cbam_diff_predict_cloud_changed, \
            cnn_bilstm_cbam_diff_predict_control_completion, \
            cnn_bilstm_cbam_diff_compensation_control_completion = \
            temp_predict(cnn_bilstm_cbam_control[i], cnn_bilstm_cbam_control_changed[i],
                         cnn_bilstm_cbam_cloud[i], cnn_bilstm_cbam_relation[i],
                         cnn_bilstm_cbam_model, cnn_bilstm_cbam_std, cnn_bilstm_cbam_mean,
                         cnn_bilstm_cbam_cloud_changed[i], cnn_bilstm_cbam_control_completion[i])

        for index in loss_indices:
            predict_diff[index].append(cnn_bilstm_cbam_diff_predict_control_completion[index])
            compensation_diff[index].append(cnn_bilstm_cbam_diff_compensation_control_completion[index])

    result = np.zeros([20, 2])
    for i in range(20):
        result[i, 0] = sum(predict_diff[i]) / len(predict_diff[i])
        result[i, 1] = sum(compensation_diff[i]) / len(compensation_diff[i])

    df = pd.DataFrame(result)
    df.columns = ['predict_diff', 'compensation_diff']
    df.to_csv('compensation_diff.csv', index=False)
