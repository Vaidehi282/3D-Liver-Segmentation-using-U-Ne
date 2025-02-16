import numpy as np
import torch
from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
from monai.losses import DiceLoss
from tqdm import tqdm


def dice_metrics(predicted, target):
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1-dice_value(predicted, target).item()
    return value

def calculate_weigths(val1, val2):
    # val 1 - background pixels
    # val2 - foreground pixels
    # if
    # val1 = 90, val2 = 10
    # w = [0.9, 0.1]
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count/summ
    
    # it inverts to get higher value to foreground [1.11, 10]
    weights = 1/weights

    # normalizes value again [0.1, 0.9] lower value to background
    summ = weights.sum()
    weights = weights/summ
    return torch.tensor(weights, dtype=torch.float32)


def train(model, data_in, loss, optim, max_epoch, model_dir, test_interval = 1, device=torch.device("cpu")):
    best_diScore = -1
    best_diScore_epoch = -1
    train_score_list = []
    test_score_list = []
    train_loss_list = []
    test_loss_list = []
    train_loader, test_loader = data_in

    for epoch in range(max_epoch):
        print('-'*10)
        print(f'Epoch: {epoch+1}/{max_epoch}')
        model.train()
        train_epoch_loss = 0  #stores total loss in epoch
        train_step = 0  # counts no. of batches
        epoch_score_train = 0

        for batch_data in train_loader:
            train_step += 1

            img = batch_data['image']
            label = batch_data['label']

            label = label != 0 #converts 1,2 into 1 and 0 reamins 0;
            img, label = (img.to(device), label.to(device))

            optim.zero_grad()
            outputs = model(img)

            train_loss = loss(outputs, label)
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss
            print(f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                    f"Train_loss: {train_loss.item():.4f}")
            train_metric = dice_metrics(outputs, label)
            epoch_score_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        train_epoch_loss = train_epoch_loss/train_step
        print(f'Epoch loss: {train_epoch_loss:.4f}')
        # train_loss_list.append(train_epoch_loss)
        # np.save(os.path.join(model_dir, 'loss_train.npy'), train_loss_list)
        train_loss_list.append(train_epoch_loss.detach().cpu().numpy())  # Convert tensor to numpy
        np.save(os.path.join(model_dir, 'loss_train.npy'), np.array(train_loss_list))  # Ensure NumPy format


        epoch_score_train = epoch_score_train/train_step
        print(f'Epoch DiScore: {epoch_score_train:.4f}')
        train_score_list.append(epoch_score_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), train_score_list)
        
        print('-'*20)

        if (epoch+1)% test_interval == 0:
            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_score_test = 0
                test_step = 0


                for test_data in test_loader:
                    test_step += 1

                    img = test_data['image']
                    label = test_data['label']
                    label = label != 0

                    img, label = (img.to(device), label.to(device))
                    
                    outputs = model(img)
                    test_loss = loss(outputs, label)
                    test_epoch_loss += test_loss

                    test_metric = dice_metrics(outputs, label)
                    epoch_score_test += test_metric

                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                # test_loss_list.append(test_epoch_loss)
                # np.save(os.path.join(model_dir, 'loss_test.npy'), test_loss_list)
                test_loss_list.append(test_epoch_loss.detach().cpu().numpy())  # Convert tensor to numpy
                np.save(os.path.join(model_dir, 'loss_test.npy'), np.array(test_loss_list))  # Ensure NumPy format

                epoch_score_test /= test_step
                print(f'test_dice_epoch: {epoch_score_test:.4f}')
                test_score_list.append(epoch_score_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), test_score_list)

                if epoch_score_test > best_diScore:
                    best_diScore = epoch_score_test
                    best_diScore_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))
                
                print(
                    f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
                    f"\nbest mean dice: {best_diScore:.4f} "
                    f"at epoch: {best_diScore_epoch}"
                )
    print(
        f"train completed, best_metric: {best_diScore:.4f} "
        f"at epoch: {best_diScore_epoch}")



