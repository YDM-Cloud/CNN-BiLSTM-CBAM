import torch

from utensils.generate_dataset import generate_dataset

if __name__ == '__main__':
    generate_dataset(
        100000,
        base_dir=r'E:\yandongming\anasys_semi_finished_proj\origin_ansys',
        n_loss_control=2)
