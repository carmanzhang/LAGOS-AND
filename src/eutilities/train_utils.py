import time

import torch
from sklearn import metrics
import numpy as np


def train(model, train_loader, criterion, optimizer, epoch, epochs, train_vector, logs_per_epoch=10,
          device=torch.device('cuda')):
    model.train()
    train_loss = 0
    num_batches = len(train_loader)
    start = time.time()
    for batch_idx, (MT, XL, XR, Y) in enumerate(train_loader):
        # HF, XL, XR, Y = HF.to(device), XL.to(device), XR.to(device), Y.to(device)
        XL, XR, Y = XL.to(device), XR.to(device), Y.to(device)
        optimizer.zero_grad()
        # output = model([HF, XL, XR])
        output = model([XL, XR])
        loss = criterion(output, Y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % (num_batches // logs_per_epoch) == 0 and batch_idx > 0:
            now = time.time()
            batch_size = len(Y)
            inputs_per_sec = ((batch_idx + 1) * batch_size) / (now - start)
            eta_min = (epochs * num_batches - (epoch - 1) * num_batches - (
                    batch_idx + 1)) * batch_size / inputs_per_sec / 60
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tInputs/s: {:.1f}\tRemaining: {:.1f} min'.format(
                epoch, batch_idx * len(Y), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(), inputs_per_sec, eta_min))

    train_loss /= len(train_loader)
    train_vector.append(train_loss)


def validate(model, test_loader, criterion, loss_vector, f1_vector=[], device=torch.device('cuda')):
    model.eval()
    val_loss = 0
    metadata = []
    prediction = torch.tensor([], device=device)
    true_labels = torch.tensor([], device=device)
    print('\nValidating...')
    with torch.no_grad():
        for (MT, XL, XR, Y) in test_loader:
            metadata.append(np.array([n.cpu().numpy() for n in MT]))
            # HF, XL, XR, Y = HF.to(device), XL.to(device), XR.to(device), Y.to(device)
            XL, XR, Y = XL.to(device), XR.to(device), Y.to(device)
            # output = model([HF, XL, XR])
            output = model([XL, XR])
            val_loss += criterion(output, Y).data.item()
            pred = output.sigmoid()
            prediction = torch.cat((prediction, pred))
            true_labels = torch.cat((true_labels, Y))

    true_label_numpy = [int(n[1]) for n in true_labels.cpu().numpy()]
    pred_label_numpy = [1 if n[1] > 0.5 else 0 for n in prediction.cpu().numpy()]
    pred_prob = [n[1] for n in prediction.cpu().numpy()]

    accuracy = metrics.accuracy_score(true_label_numpy, pred_label_numpy)
    f1_score = metrics.f1_score(true_label_numpy, pred_label_numpy)
    val_loss /= len(test_loader)
    loss_vector.append(val_loss)
    f1_vector.append(f1_score)
    print('Validation set: Average loss: {:.4f}\t Accuracy: {:.4f}\t F1-score: {:.4f}\n'.format(val_loss, accuracy,
                                                                                                f1_score))
    metadata = np.hstack(metadata)
    return metadata, true_label_numpy, pred_label_numpy, pred_prob