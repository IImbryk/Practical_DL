from torch.utils.tensorboard import SummaryWriter
import logging
import os


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
