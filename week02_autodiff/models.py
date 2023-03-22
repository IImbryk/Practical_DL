import tqdm


def train_model(model, criterion, optimizer, train_not_mnist_dataloader, test_not_mnist_dataloader, device, BATCH_SIZE):
    accs = []
    accs_train = []
    train_losses = []
    n_epochs = 100

    for epoch in tqdm.tqdm(range(n_epochs)):
        avg_loss = 0
        avg_acc = 0

        for batch in train_not_mnist_dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            predicts = model.forward(images)  # YOUR CODE

            loss_value = criterion(predicts, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            avg_loss += loss_value.item()
            accuracy = (predicts.argmax(dim=1) == labels).sum() / BATCH_SIZE
            avg_acc += accuracy

        avg_acc /= len(train_not_mnist_dataloader)
        accs_train.append(avg_acc)

        train_losses.append(avg_loss / len(train_not_mnist_dataloader))

        avg_acc = 0

        for batch in test_not_mnist_dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            predicts = model(images)

            accuracy = (predicts.argmax(dim=1) == labels).sum() / BATCH_SIZE
            avg_acc += accuracy

        avg_acc /= len(test_not_mnist_dataloader)
        accs.append(avg_acc)

        return model, accs_train, accs