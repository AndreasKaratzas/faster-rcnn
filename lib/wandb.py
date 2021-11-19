from tqdm import tqdm
args = []


import torch
from torch import optim
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from lib.engine import train

wandb.login()

config = dict(
    epochs=args.epochs,
    classes=args.num_classes,
    batch_size=args.batch_size,
    learning_rate=args.lr,
    project=args.project,
    backbone=args.backbone
)

def model_pipeline(hyperparameters, proj_name: str = 'AITrain'):

    # tell wandb to get started
    with wandb.init(project=proj_name, config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        # and use them to train model
        train(model, train_loader, criterion, optimizer, config)

        # and test its final performance
        test(model, test_loader)
    
    return model


def train(model, loader, criterion, optimizer, config):
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # run training and track with wandb
    total_batches = len(loader) * config.epochs
    # number of examples seen 
    example_ct = 0
    # number of batches seen
    batch_ct = 0
    
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):
            # get loss for single epoch
            loss = train_batch(images, labels, model, optimizer, criterion)
            # update metrics
            example_ct += len(images)
            batch_ct += 1

            # report wandb metrics every 25th epoch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def test(model, test_loader):
    model.eval()

    # run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f"Accuracy of the model on the {total} " + 
              f"test images: {100 * correct / total}%")
        
        wandb.log({"test_accuracy": correct / total})

    # save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")

