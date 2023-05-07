# import packages
import torch
import torch.nn as nn
import numpy as np
# data load
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import *
from sklearn.metrics import f1_score
# parsing arguments
import argparse
# visualization
import wandb #API key : 75860646e1cd233208820ec86a176305dec96058
from tqdm import tqdm


# argument parser
def parser_data(DEVICE):
    from model import config, CNN, RNN_LSTM, MLP
    parser = argparse.ArgumentParser(
        prog="Sentiment Analysis", description="2023 春 THU Introduction to Artificial Intelligence PA2 : Sentiment Analysis", add_help=True, allow_abbrev = True
    )
    parser.add_argument(
        "-e",
        "--epoch",
        dest = "epoch",
        type = int,
        default =30, 
        help = "epoch"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        dest = "learning_rate",
        type = float,
        default = 1e-3, # 1e-3
        help ="starting learning rate"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest  = "batch_size",
        type = int,
        default = 100,
        help = "batch size"
    )
    
    parser.add_argument(
        "-ml",
        "--max_length",
        dest="max_length",
        type=int,
        default=50,
        help="max sentence length",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="nn_model",
        type=str,
        default="CNN",
        help="neural network model , default is CNN, you can choose RNN_LSTM, MLP",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        dest = "optimizer",
        type = str,
        default = "adam",
        help= "optimizer for the neural network, default is adam, you can choose sgd, adam, adagrad, adadelta, rmsprop"
    )
    parser.add_argument(
        "-c",
        "--criterion",
        dest = "criterion",
        type = str,
        default = "cross_entropy",
        help = "criterion for the neural network, default is Cross Entropy Loss function, you can choose " 
    )
    
    parser.add_argument(
        "-s",
        "--scheduler",
        dest = "scheduler",
        type = str,
        default = "steplr",
        help = "scheduler for the neural network, default is stepLR, you can choose "
        
    )
    
    args = parser.parse_args()
    epoch = args.epoch
    learning_rate = args.learning_rate
    max_length = args.max_length
    batch_size = args.batch_size
    model = args.nn_model
    optimizer = args.optimizer
    criterion = args.criterion
    scheduler = args.scheduler


    # set model
    if model == "CNN":
        model = CNN(config).to(DEVICE)
    elif model == "RNN_LSTM":
        model = RNN_LSTM(config).to(DEVICE)
    elif model == "MLP":
        model = MLP(config).to(DEVICE)
    else :
        IOError("model not found")
        exit(1)
    # set optimizer
    if (optimizer == "sgd"):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif (optimizer == "adam"):  # default
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif (optimizer == "adagrad"):
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif (optimizer == "rmsprop"):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif (optimizer == "adadelta"):
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif (optimizer == "adamw"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif (optimizer == "sparseadam"):
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    else :
        IOError("optimizer not found")
        exit(1)

    # set criterion
    if(criterion == "cross_entropy"): # default
        criterion = nn.CrossEntropyLoss()
    elif(criterion == "mse"):
        criterion = nn.MSELoss()
    elif(criterion == "l1l"):
        criterion = nn.L1Loss()
    elif(criterion == "bce"):
        criterion = nn.BCELoss()
    elif(criterion == "bcell"):
        criterion = nn.BCEWithLogitsLoss()
    elif(criterion == "kldiv"):
        criterion = nn.KLDivLoss()
    elif(criterion == "sl1l"):
        criterion = nn.SmoothL1Loss()
    else :
        IOError("criterion not found")
        exit(1)

    # set scheduler
    scheduler = StepLR(optimizer, step_size=5)
    
    
    return epoch, learning_rate, max_length, batch_size, model,optimizer, criterion, scheduler
    
# data loader
def load_data(max_length, batch_size):
    
    from utils import vocab, s_vectors,build_dataset
    train_contents , train_labels = build_dataset("../Dataset/train.txt", vocab, max_length)
    val_contents, val_labels = build_dataset("../Dataset/validation.txt",vocab,max_length) 
    test_contents, test_labels = build_dataset("../Dataset/test.txt",vocab,max_length)
    
    # train dataset
    train_dataset = TensorDataset(
        torch.from_numpy(train_contents).type(torch.float),
        torch.from_numpy(train_labels).type(torch.long),
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    # validation dataset
    val_dataset = TensorDataset(
        torch.from_numpy(val_contents).type(torch.float),
        torch.from_numpy(val_labels).type(torch.long),
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    # test dataset
    test_dataset = TensorDataset(
        torch.from_numpy(test_contents).type(torch.float),
        torch.from_numpy(test_labels).type(torch.long),
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    
    return train_dataloader, val_dataloader, test_dataloader


# train
def train(train_dataloader):
    model.train()
    
    train_loss , train_accuracy = 0.0, 0.0
    count , correct = 0,0
    full_true = []
    full_pred = []
    for _, (sentences, labels) in enumerate(train_dataloader):
        sentences = sentences.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # forward
        optimizer.zero_grad()
        output = model(sentences)
        loss = criterion(output, labels)
        
        # backward
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        correct += (output.argmax(1) == labels).float().sum().item()
        count += len(sentences)
        full_true.extend(labels.cpu().numpy().tolist())
        full_pred.extend(output.argmax(1).cpu().numpy().tolist())
    train_loss *= batch_size
    train_loss /= len(train_dataloader.dataset)
    train_accuracy = correct / count
    
    scheduler.step()
    train_f1 = f1_score(np.array(full_true),np.array(full_pred),average = "binary")
    return train_loss, train_accuracy, train_f1
# valid and test
def valid_and_test(dataloader):
    model.eval()

    val_loss, val_acc = 0.0, 0.0
    count, correct = 0, 0
    full_true = []
    full_pred = []
    for _, (sentences, labels) in enumerate(dataloader):
        sentences, labels = sentences.to(DEVICE), labels.to(DEVICE)
        
        #forawrd
        output = model(sentences) # invokes forward()
        loss = criterion(output, labels)
        
        val_loss += loss.item()
        correct += (output.argmax(1) == labels).float().sum().item()
        count += len(sentences)
        full_true.extend(labels.cpu().numpy().tolist())
        full_pred.extend(output.argmax(1).cpu().numpy().tolist())
        
    val_loss *= batch_size
    val_loss /= len(dataloader.dataset)
    val_acc = correct / count
    f1 = f1_score(np.array(full_true), np.array(full_pred), average="binary")
    return val_loss, val_acc, f1
    
# main

if __name__ == "__main__":
    # define DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # get arguments
    EPOCHS, learning_rate, max_length, batch_size, model, optimizer,criterion,scheduler = parser_data(DEVICE)
    
    # load data
    train_dataloader, val_dataloader, test_dataloader = load_data(max_length, batch_size)
    
    wandb.init(project = "2023 Spring THU Introduction to Artifical Intelligence Sentiment Analysis (Binary Classification) PA",name=(f"{model.__name__}"), entity="hanys21",
               tags = [f"{learning_rate},{EPOCHS},{batch_size},{max_length},{type(optimizer).__name__},{type(criterion).__name__},{type(scheduler).__name__},{DEVICE}"])
    wandb.config = {"learning_rate": learning_rate, "epochs": EPOCHS, "batch_size": batch_size, "max_length":max_length,"optimizer":optimizer,"criterion":criterion,"scheduler":scheduler,"device":DEVICE}
    for each in tqdm(range(1, EPOCHS + 1)):
        tr_loss, tr_acc, tr_f1 = train(train_dataloader)
        val_loss, val_acc, val_f1 = valid_and_test(val_dataloader)
        test_loss, test_acc, test_f1 = valid_and_test(test_dataloader)
        # logging
        wandb.log(
            {
                "Train Loss": tr_loss,
                "Train Accuracy": tr_acc,
                "Train f1": tr_f1,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc,
                "Validation f1": val_f1,
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
                "Test f1": test_f1,
            }
        )
        print(
            f"for epoch {each}/{EPOCHS},Train Accuracy: {tr_acc:.4f},Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f} (in average)"
        )
    
    wandb.finish()