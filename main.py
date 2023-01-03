import argparse
import sys
import pandas as pd
import torch
import click

from data import mnist
from model import MyAwesomeModel
from torch import nn

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel(784, 10, [512, 256, 128])
    train_set, _ = mnist()

    print(train_set.shape())
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 10
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in train_set:
            steps += 1
            
            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # if steps % print_every == 0:
            #     # Model in inference mode, dropout is off
            #     model.eval()
                
            #     # Turn off gradients for validation, will speed up inference
            #     with torch.no_grad():
            #         test_loss, accuracy = validation(model, testloader, criterion)
                
            #     print("Epoch: {}/{}.. ".format(e+1, epochs),
            #           "Training Loss: {:.3f}.. ".format(running_loss/print_every),
            #           "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
            #           "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
            #     running_loss = 0
                
            #     # Make sure dropout and grads are on for training
            #     model.train()


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
