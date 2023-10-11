import torch
import numpy as np
import os
from modules.cnn_bilstm_cbam import CNN_BiLSTM_CBAM
from modules.cnn_bilstm_sa import CNN_BiLSTM_SA
from modules.cnn_gru import CNN_GRU
from modules.cnn_gru_cbam import CNN_GRU_CBAM
from modules.cnn_lstm_cbam import CNN_LSTM_CBAM
from modules.cnn_lstm_sa import CNN_LSTM_SA


def predict(model_file, data_folder, data_idx):
    # load model
    Model = os.path.basename(model_file)[:-3]
    model = eval(f'{Model}(device="cpu")')
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
    # load dataset
    data_folder += '/'
    control = torch.load(data_folder + 'control.pt')[data_idx]
    control_changed = torch.load(data_folder + 'control_changed.pt')[data_idx]
    cloud = torch.load(data_folder + 'cloud.pt')[data_idx]
    relation = torch.load(data_folder + 'relation.pt')[data_idx]
    cloud_changed = torch.load(data_folder + 'cloud_changed.pt')[data_idx]
    control_completion = torch.load(data_folder + 'control_completion.pt')[data_idx]
    mean = torch.load(data_folder + 'mean.pt').numpy()
    std = torch.load(data_folder + 'std.pt').numpy()
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
    train_cloud = cloud.numpy() * std + mean
    train_relation = relation.numpy() * std + mean
    train_control = control.numpy() * std + mean
    train_control_changed = control_changed.numpy() * std + mean
    train_control_completion = control_completion.numpy() * std + mean
    compensation_control_completion = compensation(train_control_changed, predict_control_completion)
    diff_predict_cloud_changed = np.linalg.norm(predict_cloud_changed - train_cloud_changed, axis=1)
    diff_predict_control_completion = np.linalg.norm(predict_control_completion - train_control_completion, axis=1)
    diff_compensation_control_completion = np.linalg.norm(compensation_control_completion - train_control_completion,
                                                          axis=1)
    diff_cloud_changed = np.linalg.norm(train_cloud - train_cloud_changed, axis=1)
    diff_control_changed = np.linalg.norm(train_control - train_control_changed, axis=1)
    return train_control, train_control_changed, train_cloud, train_relation, \
        train_cloud_changed, train_control_completion, \
        predict_cloud_changed, diff_predict_cloud_changed, \
        predict_control_completion, diff_predict_control_completion, \
        compensation_control_completion, diff_compensation_control_completion, \
        diff_control_changed, diff_cloud_changed


def compensation(real_incomplete_data, predict_complete_data, nan=0):
    nan_condition = real_incomplete_data.all(1) == nan
    if len(np.where(nan_condition)[0]) == 0:
        return real_incomplete_data

    mean = predict_complete_data[~nan_condition].mean(axis=0)
    std = predict_complete_data[~nan_condition].std(axis=0)
    predict_complete_normalized_data = predict_complete_data.copy()
    predict_complete_normalized_data = (predict_complete_normalized_data - mean) / std

    mean = real_incomplete_data[~nan_condition].mean(axis=0)
    std = real_incomplete_data[~nan_condition].std(axis=0)
    real_complete_data = real_incomplete_data.copy()
    real_complete_data[nan_condition] = predict_complete_normalized_data[nan_condition] * std + mean

    return real_complete_data
