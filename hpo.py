import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse


def test(model, test_loader, criterion, device, args):
    model.eval()
    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(predictions == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    print(
        f"Test for Hyperparameters, lr: {args.lr}, "
        f"batch size: {args.batch_size}")

    print(f"Testing Loss: {total_loss}, "
          f"Testing Accuracy: {100 * total_acc}")


def train(model, train_loader, valid_loader, criterion, optimizer, device):
    loss_counter = 0
    best_valid_loss = float('inf')
    epochs = 3
    samples = 0
    data_loaders = {'train': train_loader, 'valid': valid_loader}

    for epoch in range(epochs):
        print(f"Epoch:{epoch}")
        for phase in ['train', 'valid']:
            print(f"Phase:{phase}")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, predictions = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)
                samples += len(inputs)

                # NOTE: Comment lines below to train and test on whole dataset
                if samples > (0.2 * len(data_loaders[phase].dataset)):
                    break

            epoch_loss = running_loss // len(data_loaders[phase])
            epoch_acc = running_corrects // len(data_loaders[phase])

            if phase == 'valid':
                if epoch_loss < best_valid_loss:
                    best_valid_loss = epoch_loss
                else:
                    loss_counter += 1

            print('{} loss: {:.3f}, accuracy: {:.2f}, best valid loss: {:.3f}'
                  .format(phase, epoch_loss, epoch_acc, best_valid_loss))

        if loss_counter == 1:
            break
        if epoch == 0:
            break

    return model


def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        n_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_features, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 133))

    return model


def create_data_loaders(data, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(data, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f'Hyperparameters: {args.lr}, Batch Size: {args.batch_size}')
    print(f'Data paths: {args.data}')

    train_loader = create_data_loaders(os.environ['SM_CHANNEL_TRAIN'], args.batch_size)
    validation_loader = create_data_loaders(os.environ['SM_CHANNEL_VALID'], args.batch_size)
    test_loader = create_data_loaders(os.environ['SM_CHANNEL_TEST'], args.batch_size)

    model = net()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr)

    print("Training started......")
    model = train(model, train_loader, validation_loader, criterion, optimizer, device)

    print("Testing started......")
    test(model, test_loader, criterion, device, args)

    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input training batch size (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs (default : 3)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)")

    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()

    main(args)
