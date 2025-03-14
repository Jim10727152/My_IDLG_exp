#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from models.Nets import CIFAR_CNN, MLP, LeNet, CIFAR_MLP
from DP_noise.Laplace import add_laplace_noise_to_gradients, add_staircase_noise_to_gradients, add_laplace_noise_to_first_layer


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


def main():
    automode = True
    if automode == True :
        name_of_dataset = 'MNIST'
        model_type = 'MLP'
        dp_noise_mechanism = 'laplace'
        epsilon = 1
    else:
        name_of_dataset = input("Choose dataset (MNIST/CIFAR10): ").strip().upper()  # choose the dataset
        model_type = input('Please choose the model you like (MLP/CNN): ').strip().upper()
        dp_noise_mechanism = input( 'Choose the mechanism of differential privacy( laplace / staircase ) : ')        
        epsilon = float( input( 'Enter the epsilon ( -1 for no dp noise ) : '))

    root_path = '.'
    data_path = os.path.join(root_path, './data').replace('\\\\', '/')

    lr = 1.0
    Iteration = 300
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if name_of_dataset == 'MNIST':
        tt = transforms.Compose([transforms.ToTensor()])
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dataset = datasets.MNIST(data_path, download=True, transform=tt)
    elif name_of_dataset == 'CIFAR10':
        tt = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        shape_img = (32, 32)
        num_classes = 10
        channel = 3
        hidden = 128 * 4 * 4
        dataset = datasets.CIFAR10(data_path, download=True, transform=tt)
    else:
        print("Error: Unsupported dataset!")
        exit(0)


    if model_type == 'CNN' and name_of_dataset == 'MNIST':
        net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes).to(device)
        net.apply(weights_init)
    elif model_type == 'MLP' and name_of_dataset == 'MNIST':
        net = MLP().to(device)
    elif model_type == 'CNN' and name_of_dataset == 'CIFAR10':   
        net = CIFAR_CNN(num_classes=num_classes).to(device)
        Iteration = 1000
    elif model_type == 'MLP' and name_of_dataset == 'CIFAR10':
        net = CIFAR_MLP(num_classes=num_classes).to(device)
        Iteration = 1000
        lr = 0.5
    else:
        print('\nError: Unrecognized model or incompatible dataset!\n')
        exit(0)

    criterion = nn.CrossEntropyLoss().to(device)
    idx_shuffle = np.random.permutation(len(dataset))
    idx = idx_shuffle[0]

    # Prepare ground-truth data
    gt_data = dataset[idx][0].unsqueeze(0).to(device)
    gt_label = torch.Tensor([dataset[idx][1]]).long().to(device)

    # Compute original gradient
    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    print( '\n\nStart to add differential privacy noise, mechanism  : ', dp_noise_mechanism )
    if epsilon == -1:
        noisy_dy_dx = original_dy_dx
    else:
        if dp_noise_mechanism == 'laplace':
            noisy_dy_dx = add_laplace_noise_to_first_layer(original_dy_dx, epsilon)
        elif dp_noise_mechanism == 'staircase':
            noisy_dy_dx = add_staircase_noise_to_gradients(original_dy_dx, epsilon)
        else :
            print( 'There is no differential privacy mechanism : ', dp_noise_mechanism, '\nPlease try again ! ' )
            exit(0)

    print( 'Finish of  differenital privacy ! \n\n' )

    # Predict the ground-truth label (iDLG-specific)
    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
    print(f"Predicted label by iDLG: {label_pred.item()}")

    # Generate dummy data
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data], lr=lr)

    # 保存上一回合的 dummy_data
    last_valid_dummy_data = dummy_data.clone().detach()

    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = criterion(pred, label_pred)  # Use predicted label
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, noisy_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        try:
            optimizer.step(closure)
            loss_value = closure().item()

            # 如果 loss 是 NaN，跳出迴圈並使用上一回合的 dummy_data
            if torch.isnan(torch.tensor(loss_value)):
                print(f"Loss became NaN at iteration {iters}. Using last valid dummy data...")
                dummy_data = last_valid_dummy_data  # 恢復到上一回合的數據
                break

            # 保存當前有效的 dummy_data
            last_valid_dummy_data = dummy_data.clone().detach()

        except Exception as e:
            print(f"Error occurred during optimization: {e}")
            break

        if iters % 50 == 0:
            print(f"Iteration {iters}, Loss: {loss_value}")

    mse = F.mse_loss(dummy_data.detach(), gt_data.detach())
    print(f"\nMean Squared Error (MSE) between ground truth and dummy data: {mse.item()}")

    # Visualization
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(gt_data[0].permute(1, 2, 0).cpu() if channel == 3 else gt_data[0][0].cpu(), cmap='gray' if channel == 1 else None)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Dummy Data")
    plt.imshow(dummy_data[0].detach().permute(1, 2, 0).cpu() if channel == 3 else dummy_data[0][0].detach().cpu(), cmap='gray' if channel == 1 else None)
    plt.axis('off')

    # 保存圖片到 results 資料夾
    save_dir = os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True) or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(save_dir, exist_ok=True)  # 如果資料夾不存在則創建
    img_name = 'Defense_iDLG_attack_' + str(dp_noise_mechanism) + '_epsilon_' +str(epsilon) + '_' + str(random.randint(0, 1000000)) + '.png'
    save_path = os.path.join(save_dir, img_name)
    plt.savefig(save_path, bbox_inches='tight')  
    print(f"Save the image to : {save_path}")
    # show the image
    plt.show()
    plt.close()  


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time 
    print(f"Execution time of A(): {execution_time:.4f} seconds")