from torch.utils.tensorboard import SummaryWriter
import logging
import os
import torch
import copy
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm

def log(path, file):
    """[Create a log file to record the experiment's logs]

    Arguments:
        path {string} -- path to the directory
        file {string} -- file name

    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, device, n_epochs, batch_size):

    logger = log(path="logs/", file="test.logs")
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

        logger.info("Epoch {}".format(epoch))

        logger.info("The mean accuracy train: {:.3f}".format(avg_acc_train))
        logger.info("The mean accuracy test: {:.3f}".format(avg_acc_test))
        logger.info("-------------------------------")

        tb.add_scalar("Loss", avg_loss_train, epoch)
        tb.add_scalar("Accuracy_train", avg_acc_train, epoch)
        tb.add_scalar("Accuracy_test", avg_acc_test, epoch)

        # tb.add_histogram("conv1.bias", model.conv1.bias, epoch)
        # tb.add_histogram("conv1.weight", model.conv1.weight, epoch)
        # tb.add_histogram("conv2.bias", model.conv2.bias, epoch)
        # tb.add_histogram("conv2.weight", model.conv2.weight, epoch)

        # print("epoch:", epoch, "accuracy:", avg_acc_train, "loss:", avg_loss_train)

    return model, optimizer
## return model, optimizer (params),

## set random state for train/test


## random state could be chage on different PC. Save index or

def train_model_with_scheduler(model, loss, optimizer, train_dataloader, test_dataloader, scheduler, num_epochs, device, batch_size):

    best_acc = 0
    tb = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)
        gt = []
        net_outputs = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = test_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                    else:
                        gt.extend(labels.data.cpu().numpy())
                        net_outputs.extend(preds_class.data.cpu().numpy())

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            if phase == 'train':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)
                scheduler.step()
            else:
                bacc = balanced_accuracy_score(gt, net_outputs)
                print('{} Loss: {:.4f}, balanced_accuracy_score: {:.4f}, accuracy_score: {:.4f}'.format(phase, epoch_loss, bacc, accuracy_score(gt, net_outputs)), flush=True)

            if phase == 'val' and bacc >= best_acc:
                best_acc = bacc
                best_model_wts = copy.deepcopy(model.state_dict())

            tb.add_scalar("Loss", epoch_loss, epoch)
            tb.add_scalar("Accuracy_train", epoch_acc, epoch)
            tb.add_scalar("Accuracy_test", accuracy_score(gt, net_outputs), epoch)

    model.load_state_dict(best_model_wts)
    return model, best_acc
