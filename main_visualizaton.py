from utensils.visualization_handler import show_loss_rmse, show_predict_diff

if __name__ == '__main__':
    show_loss_rmse()

    show_predict_diff(
        '2023-08-01_07-06-48-020800_n_train(100000)n_control(20)n_cloud(1208)point_only(0)n_loss_control(2,0.5)standardization(1)',
        'n_train(100000)n_control(20)n_cloud(1208)point_only(0)n_loss_control(2,0.5)standardization(1)',
        0)
