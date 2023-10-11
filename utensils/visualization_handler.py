import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import netron
from .onnx_handler import export_model
import os
from .predict_handler import predict
import pandas as pd


def relation_to_polydata(relation):
    polydata = pv.PolyData()
    for i in range(relation.shape[0]):
        triangle = pv.Triangle(relation[i, :].reshape([-1, 3]))
        polydata += triangle
    return polydata


def show_predict_diff(model_file, data_folder, data_idx):
    pt_file = [file for file in os.listdir(f'result/{model_file}') if file.endswith('.pt')][0]
    model_file = f'result/{model_file}/{pt_file}'
    data_folder = f'data/generate_dataset/{data_folder}'
    origin_control = np.loadtxt(r'data/origin_geometry/control positions.txt', delimiter=',')
    # predict
    train_control, train_control_changed, train_cloud, train_relation, \
        train_cloud_changed, train_control_completion, \
        predict_cloud_changed, diff_predict_cloud_changed, \
        predict_control_completion, diff_predict_control_completion, \
        compensation_control_completion, diff_compensation_control_completion, \
        diff_control_changed, diff_cloud_changed = predict(model_file, data_folder, data_idx)
    # visualization parameters
    point_size = 10
    opacity = 0.2
    control_loss_indices = (train_control == np.zeros(3)).all(axis=1)
    loss_opacity = np.ones(train_control.shape[0])
    loss_opacity[control_loss_indices] = opacity
    train_control = pv.PolyData(origin_control)
    train_control_changed = pv.PolyData(train_control_completion)
    train_control_changed.point_data['diff'] = diff_control_changed
    train_cloud = pv.PolyData(train_cloud)
    train_cloud = train_cloud.delaunay_3d()
    train_relation = relation_to_polydata(train_relation)
    train_cloud_changed = pv.PolyData(train_cloud_changed)
    train_cloud_changed = train_cloud_changed.delaunay_3d()
    train_cloud_changed.point_data['diff'] = diff_cloud_changed
    predict_cloud_changed = pv.PolyData(predict_cloud_changed)
    predict_cloud_changed = predict_cloud_changed.delaunay_3d()
    predict_cloud_changed.point_data['diff'] = diff_predict_cloud_changed
    predict_control_completion = pv.PolyData(predict_control_completion)
    predict_control_completion.point_data['diff'] = diff_predict_control_completion
    compensation_control_completion = pv.PolyData(compensation_control_completion)
    compensation_control_completion.point_data['diff'] = diff_compensation_control_completion

    # show one by one
    p = pv.Plotter(title='train_control')
    p.set_background('w')
    p.add_mesh(train_cloud, color='tan', opacity=opacity)
    p.add_points(train_control, color='r', opacity=loss_opacity, render_points_as_spheres=True, point_size=point_size)
    p.show()
    p = pv.Plotter(title='train_cloud')
    p.set_background('w')
    p.add_mesh(train_cloud, color='tan')
    p.show()
    p = pv.Plotter(title='train_relation')
    p.set_background('w')
    p.add_mesh(train_relation, color='tan')
    p.show()
    p = pv.Plotter(title='train_control_changed')
    p.set_background('w')
    p.add_mesh(train_cloud_changed, color='tan', opacity=opacity)
    p.add_points(train_control_changed, color='r', render_points_as_spheres=True, point_size=point_size,
                 opacity=loss_opacity, scalars='diff', cmap='jet', scalar_bar_args={'color': 'black'})
    p.show()
    p = pv.Plotter(title='train_cloud_changed')
    p.set_background('w')
    p.add_mesh(train_cloud_changed, color='tan', scalars='diff', cmap='jet', scalar_bar_args={'color': 'black'})
    p.show()
    p = pv.Plotter(title='predict_cloud_changed')
    p.set_background('w')
    p.add_mesh(predict_cloud_changed, cmap='jet', scalars='diff', scalar_bar_args={'color': 'black'})
    p.add_mesh(predict_cloud_changed, cmap='jet', scalars='diff', scalar_bar_args={'color': 'black'}, clim=[0, 0.92])

    p.show()
    p = pv.Plotter(title='predict_control_completion')
    p.set_background('w')
    p.add_points(predict_control_completion, cmap='jet', scalars='diff', scalar_bar_args={'color': 'black'},
                 render_points_as_spheres=True, point_size=point_size)
    p.show()
    p = pv.Plotter(title='compensation_control_completion')
    p.set_background('w')
    p.add_points(compensation_control_completion, cmap='jet', scalars='diff', scalar_bar_args={'color': 'black'},
                 render_points_as_spheres=True, point_size=point_size)
    p.show()


def show_loss_rmse(result_folders=None, legend_names=None):
    suptitles = ['train loss', 'test loss', 'train RMSE', 'test RMSE']
    plot_results = []
    module_names = []
    if result_folders is None:
        result_folders = os.listdir('result')
    n_result = len(result_folders)
    for result_folder in result_folders:
        result_folder = f'result/{result_folder}'
        plot_name = [file for file in os.listdir(result_folder) if file.startswith('plot_result')][0]
        plot_result = pd.read_csv(f'{result_folder}/{plot_name}', header=0).values
        plot_results.append(plot_result)
        if legend_names is None:
            module_name = [file for file in os.listdir(f'{result_folder}') if file.endswith('.pt')][0]
            module_names.append(module_name[:-3])
    if legend_names is not None:
        module_names = legend_names
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(suptitles[i])
        [plt.plot(plot_results[j][:, i], label=module_names[j]) for j in range(n_result)]
        plt.legend()
    plt.show()

# def show_diff_box(data_folder, result_folders=None):
#     if result_folders is None:
#         result_folders = os.listdir('result')
#     statistics_diff_box(data_folder, 'result', result_folders)
#     all_diff = None
#     labels = []
#     for r, result_folder in enumerate(result_folders):
#         # load diff
#         diff = pd.read_csv(f'result/{result_folder}/predict_diff.csv', header=None).values
#         if r == 0:
#             all_diff = np.zeros([diff.shape[0], len(result_folders)])
#         all_diff[:, r] = diff.flatten()
#         # append label
#         module_file = [file for file in os.listdir(f'result/{result_folder}') if file.endswith('.pt')][0]
#         labels.append(module_file[:-3])
#     plt.boxplot(all_diff, labels=labels)
#     plt.show()


# def show_net_frame(result_folder):
#     if 'net_frame.onnx' not in os.listdir(result_folder):
#         export_model(result_folder)
#     netron.start(f'{result_folder}/net_frame.onnx')
