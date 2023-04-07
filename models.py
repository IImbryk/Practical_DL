import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, device, n_epochs, batch_size):

    tb = SummaryWriter()

    for epoch in range(n_epochs):
        avg_loss_train = 0
        avg_acc_train = 0

        for batch in train_dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            predicts = model.forward(images)
            loss_value = criterion(predicts, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            avg_loss_train += loss_value.item()
            accuracy = (predicts.argmax(dim=1) == labels).sum() / batch_size
            avg_acc_train += accuracy.item()

        # avg_loss_train /= len(train_dataloader)
        avg_acc_train /= len(train_dataloader)

        avg_acc_test = 0

        for batch in test_dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            predicts = model(images)

            accuracy = (predicts.argmax(dim=1) == labels).sum() / batch_size
            avg_acc_test += accuracy

        avg_acc_test /= len(test_dataloader)

        # add callback -- with format !

        tb.add_scalar("Loss", avg_loss_train, epoch)
        tb.add_scalar("Accuracy_train", avg_acc_train, epoch)
        tb.add_scalar("Accuracy_test", avg_acc_test, epoch)

        # tb.add_histogram("conv1.bias", model.conv1.bias, epoch)
        # tb.add_histogram("conv1.weight", model.conv1.weight, epoch)
        # tb.add_histogram("conv2.bias", model.conv2.bias, epoch)
        # tb.add_histogram("conv2.weight", model.conv2.weight, epoch)

        # print("epoch:", epoch, "accuracy:", avg_acc_train, "loss:", avg_loss_train)

    return model
## return model, optimizer (params),

## set random state for train/test

## random state could be chage on different PC. Save index or
