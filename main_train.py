from utensils.train_handler import run
from modules.cnn_bilstm_cbam import CNN_BiLSTM_CBAM
from modules.cnn_bilstm_sa import CNN_BiLSTM_SA
from modules.cnn_gru import CNN_GRU
from modules.cnn_gru_cbam import CNN_GRU_CBAM
from modules.cnn_lstm_cbam import CNN_LSTM_CBAM
from modules.cnn_lstm_sa import CNN_LSTM_SA
from modules.cnn_bilstm_cbam_without_relation import CNN_BiLSTM_CBAM_Without_Relation

if __name__ == '__main__':
    # train
    data_folder = 'n_train(100000)n_control(50)n_cloud(1000)point_only(1)n_loss_control(2,0.5)standardization(1)'
    run(CNN_BiLSTM_CBAM, data_folder)
    run(CNN_BiLSTM_SA, data_folder)
    run(CNN_GRU, data_folder)
    run(CNN_GRU_CBAM, data_folder)
    run(CNN_LSTM_CBAM, data_folder)
    run(CNN_LSTM_SA, data_folder)
    run(CNN_BiLSTM_CBAM_Without_Relation, data_folder)
