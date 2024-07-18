import numpy as np
import torch
import random
import os
from tqdm import tqdm
import argparse
import WBANet
from myutils.create_dataset import getImage, get_train_loader, createDataset
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from myutils.evaluate_function import evaluate, postprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model: WBANet, args, im1, im2, im_gt, im_di, patch_size, preclassify_lab):
    train_loader = get_train_loader(args, im1, im2, im_gt, im_di, patch_size, preclassify_lab)
    lr = args.lr
    gamma = args.gamma
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    # model.load_state_dict(torch.load("./checkpoints/model_lastyellowriver.pth"))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.stepsize, gamma=gamma)
    best_acc = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            # print(label)
            # print(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        if epoch_accuracy > best_acc:
            best_acc = epoch_accuracy
            checkpoint_best = "./checkpoints/model_best" + str(args.dataset) + ".pth"
            torch.save(model.state_dict(), checkpoint_best)
        if args.isScheduler:
            scheduler.step()
        checkpoint_last = "./checkpoints/model_last" + str(args.dataset) + ".pth"
        torch.save(model.state_dict(), checkpoint_last)

        print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")


if __name__ == '__main__':
    seed = 42
    seed_everything(seed)
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='yellowriver', type=str)
    parser.add_argument("--batchsize", default=128, type=int)
    parser.add_argument("--epochs", default=30, type=int, help="yellow, sulz, chao try 30, 15/10, 25, respectively")
    parser.add_argument("--lr", default=1e-3, type=float, help="yellow, sulz, chao try 1e-3, 5e-3, 1e-4, respectively")
    parser.add_argument("--gamma", default=0.75, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--patchsize", default=8, type=int, help="patch size must be even")
    parser.add_argument("--isScheduler", default=False, type=bool)
    parser.add_argument("--stepsize", default=5, type=int)
    args = parser.parse_args()

    im1, im2, im_gt, im_di, patch_size, preclassify_lab = getImage(args)

    model = WBANet.WBANet(args).to(device)

    train(model, args, im1, im2, im_gt, im_di, patch_size, preclassify_lab)

    checkpoint_path = "./checkpoints/model_best" + str(args.dataset) + ".pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    ylen, xlen = im_di.shape
    _, _, x_test = createDataset(args, im1, im2, im_gt, im_di, patch_size, preclassify_lab)
    outputs = np.zeros((ylen, xlen))
    for i in range(ylen):
        for j in range(xlen):
            if preclassify_lab[i, j] != 1.5:
                outputs[i, j] = preclassify_lab[i, j]
            else:
                img_patch = x_test[i * xlen + j, :, :, :]
                img_patch = img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2])
                img_patch = torch.FloatTensor(img_patch).to(device)
                prediction = model(img_patch)
                # print(prediction)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i, j] = prediction + 1

        if (i + 1) % 50 == 0:
            print('... ... row', i + 1, ' handling ... ...')

    outputs = outputs - 1
    res = outputs * 255
    res = postprocess(res)
    evaluate(im_gt, res)
    plt.imshow(res, 'gray')
    plt.axis('off')  # remove coordinate axis
    plt.xticks([])  # remove x axis
    plt.yticks([])  # remove y axis
    plt.show()
