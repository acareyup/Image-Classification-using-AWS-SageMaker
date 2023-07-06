import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import importlib
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, device, hook, args):
    smd = importlib.import_module('smdebug')
    modes = getattr(smd, 'modes')
    model.eval()
    hook.set_mode(modes.EVAL)
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


def train(model, train_loader, validation_loader, loss_criterion, optimizer, device, hook):
    smd = importlib.import_module('smdebug')
    loss_counter = 0
    best_loss = 1e6
    epochs = 30
    hook.set_mode(smd.modes.TRAIN)  # set debugging hook
    image_dataset = {'train': train_loader, 'valid': validation_loader}

    for epoch in range(epochs):
        print(f"Epoch:{epoch}")
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)  # set debugging hook
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1

            print('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                           epoch_loss,
                                                                           epoch_acc,
                                                                           best_loss))
        if loss_counter == 1:
            break
        if epoch == 0:
            break
    return model


def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


def save_model(model, model_dir):
    print(f"Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        nfeatures = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(nfeatures, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256),
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
    pytorch = importlib.import_module('smdebug.pytorch')

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

    hook = pytorch.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(criterion)
    print("Training started......")
    model = train(model, train_loader, validation_loader, criterion, optimizer, device, hook)

    print("Testing started......")
    test(model, test_loader, criterion, device, hook, args)

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
        default=30,
        metavar="N",
        help="number of epochs (default : 30)"
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
