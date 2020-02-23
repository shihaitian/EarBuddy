# -*- coding: utf-8 -*-
from utils import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import utils
import pickle
from ml_dataloader import *
import sklearn.metrics as metrics
import json
import logging
from copy import deepcopy
from ml_model import *
import scipy
import time

######################################
# train auto encoder-decoder, then freeze encoder and train a fc
######################################

## STEP 1 train encoder-decoder

def train_reconstruction_two_models(model1, model2, device, train_loader, criterion, optimizer_encoder, optimizer_decoder, epoch):
    model1.train()
    model2.train()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        output = model1(data)
        output = model2(output)
        loss = criterion(output, data)
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    logging.info('Train Epoch: {0}\tLoss: {1:.6f}'.format(
        epoch,
        train_loss
        )
    )
    return {"loss":train_loss}

def test_reconstruction_two_models(model1, model2, device, test_loader, function):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0

    test_list = []
    pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model1(data)
            output = model2(output)
            test_loss += function(output, data).item() # sum up batch loss

            if (len(test_list) == 0):
                test_list = deepcopy(data.to("cpu").data.numpy())
                pred_list = deepcopy(output.to("cpu").data.numpy())

    test_loss /= len(test_loader)

    return {"loss": test_loss, "true_label": test_list, "pred_label": pred_list}

def trainandsave_auto_encoderdecoder(pytorch_encoder_model, pytorch_decoder_model, train_loader, test_loader, logfile):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(device)

    encoder_model = pytorch_encoder_model.to(device)
    decoder_model = pytorch_decoder_model.to(device)

    early_stop_encoder = EarlyStopping(name = encoder_model.getname())
    early_stop_decoder = EarlyStopping(name = decoder_model.getname())
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # lossfunction = nn.functional.cross_entropy
    lossfunction = nn.functional.mse_loss

    lr = 0.001
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer_encoder = optim.SGD(encoder_model.parameters(), lr=lr, momentum = 0.9)
    # optimizer_decoder = optim.SGD(decoder_model.parameters(), lr=lr, momentum = 0.9)

    optimizer_encoder = optim.Adam(encoder_model.parameters(), lr=lr)
    optimizer_decoder = optim.Adam(decoder_model.parameters(), lr=lr)

    # train
    train_error = []
    test_error = []
    for epoch in range(80):
        print(epoch)
        if (epoch % 5 == 4):
            lr /= 3
            # optimizer_encoder = optim.SGD(encoder_model.parameters(), lr=lr, momentum = 0.9)
            # optimizer_decoder = optim.SGD(decoder_model.parameters(), lr=lr, momentum = 0.9)
            optimizer_encoder = optim.Adam(encoder_model.parameters(), lr=lr)
            optimizer_decoder = optim.Adam(decoder_model.parameters(), lr=lr)
        
        train_results = train_reconstruction_two_models(encoder_model, decoder_model, device, train_loader, criterion, optimizer_encoder, optimizer_decoder, epoch)
        test_results = test_reconstruction_two_models(encoder_model, decoder_model, device, test_loader, lossfunction)
        train_error.append(train_results["loss"])
        test_error.append(test_results["loss"])
        with open(logfile.replace(".log", "_loss.txt"), "w") as f:
            f.write("\n".join([str(x) + "," + str(y) for x, y in zip(train_error, test_error)]))
        early_stop_encoder(test_results["loss"], encoder_model)
        early_stop_decoder(test_results["loss"], decoder_model)
        
        if early_stop_decoder.early_stop:
            print("Early stopping")
            break

    with open("final_img_test.pkl", "wb") as f:
        pickle.dump(test_results,f)

    logging.info('Finished Training!')
    # save model
    encoder_model.load_state_dict(torch.load(early_stop_encoder.model_filename))
    encoder_model_name = encoder_model.getname()
    torch.save(encoder_model, encoder_model_name + '.pkl')
    torch.save(encoder_model.state_dict(), encoder_model_name + '_params.pkl')

    decoder_model.load_state_dict(torch.load(early_stop_decoder.model_filename))
    decoder_model_name = decoder_model.getname()
    torch.save(decoder_model, decoder_model_name + '.pkl')
    torch.save(decoder_model.state_dict(), decoder_model_name + '_params.pkl')

## STEP 2: train fc

def train_classification_second_models(model1, model2, device, train_loader, criterion, optimizer2, epoch):
    model2.train()
    train_loss = 0
    for param in model1.parameters():
        param.requires_grad = False

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer2.zero_grad()
        output = model1(data)
        output = model2(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer2.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    logging.info('Train Epoch: {0}\tLoss: {1:.6f}'.format(
        epoch,
        train_loss
        )
    )
    return {"loss":train_loss}

def test_classification_second_models(model1, model2, device, test_loader, function):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0

    test_list = []
    pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model1(data)
            output = model2(output)
            test_loss += function(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            test_list += list(target.to("cpu").data.numpy().reshape(-1))
            pred_list += list(pred.to("cpu").data.numpy().reshape(-1))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct, 
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
        )
    )

    logging.info(metrics.classification_report(y_true = test_list,
        y_pred = pred_list,
        target_names  = gesture_config["train_labels_no_noise"]))

    cm = metrics.confusion_matrix(y_true = test_list,
        y_pred = pred_list)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    string = "\n"
    for row in cm:
        string += "  ".join(["{:.3f}".format(x) for x in row]) + "\n" 
    logging.info(string)

    return {"loss": test_loss, "true_label": test_list, "pred_label": pred_list}

def trainandsave_encoder_classification(pytorch_encoder_model, pytorch_fcclassification_model, train_loader, test_loader, logfile):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(device)

    encoder_model = torch.load(pytorch_encoder_model.getname() + ".pkl")
    encoder_model = encoder_model.to(device)
    for param in encoder_model.parameters():
        param.requires_grad = False

    fc_model = pytorch_fcclassification_model.to(device)

    early_stop_fc = EarlyStopping(name = fc_model.getname())
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    lossfunction = nn.functional.cross_entropy
    # lossfunction = nn.functional.mse_loss

    lr = 0.0004
    # optimizer_decoder = optim.SGD(fc_model.parameters(), lr=lr, momentum = 0.9)

    optimizer_fc = optim.Adam(fc_model.parameters(), lr=lr)

    # train
    train_error = []
    test_error = []
    for epoch in range(400):
        print(epoch)
        if (epoch % 20 == 19):
            lr /= 2
            # optimizer_fc = optim.SGD(fc_model.parameters(), lr=lr, momentum = 0.9)
            optimizer_fc = optim.Adam(fc_model.parameters(), lr=lr)
        
        train_results = train_classification_second_models(encoder_model, fc_model, device, train_loader, criterion, optimizer_fc, epoch)
        test_results = test_classification_second_models(encoder_model, fc_model, device, test_loader, lossfunction)
        train_error.append(train_results["loss"])
        test_error.append(test_results["loss"])
        with open(logfile.replace(".log", "_loss.txt"), "w") as f:
            f.write("\n".join([str(x) + "," + str(y) for x, y in zip(train_error, test_error)]))
        early_stop_fc(test_results["loss"], fc_model)
        
        if early_stop_fc.early_stop:
            print("Early stopping")
            break

    logging.info('Finished Training!')
    # save model
    fc_model.load_state_dict(torch.load(early_stop_fc.model_filename))
    fc_model_name = fc_model.getname()
    torch.save(fc_model, fc_model_name + '.pkl')
    torch.save(fc_model.state_dict(), fc_model_name + '_params.pkl')

######################################
# train auto encoder-decoder, and a fc together
######################################

def train_reconstruction_three_models(model1, model2, model_fc, device, train_loader, criterion_reconstruction, criterion_classification, 
    optimizer_encoder, optimizer_decoder, optimizer_fc, epoch):
    model1.train()
    model2.train()
    model_fc.train()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        optimizer_fc.zero_grad()

        output = model1(data)
        reconstruction = model2(output)
        pred = model_fc(reconstruction)
        loss = 1 * criterion_reconstruction(reconstruction, data) + criterion_classification(pred, target)
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()
        optimizer_fc.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    logging.info('Train Epoch: {0}\tLoss: {1:.6f}'.format(
        epoch,
        train_loss
        )
    )
    return {"loss":train_loss}

def test_reconstruction_three_models(model1, model2, model_fc, device, test_loader, function_reconstruction, function_classification):
    model1.eval()
    model2.eval()
    model_fc.eval()
    test_loss = 0
    correct = 0

    test_list = []
    pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model1(data)
            reconstruction = model2(output)
            pred = model_fc(reconstruction)
            test_loss += 1 * function_reconstruction(reconstruction, data).item() + function_classification(pred, target).item()
            
            pred_ = pred.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            test_list += list(target.to("cpu").data.numpy().reshape(-1))
            pred_list += list(pred_.to("cpu").data.numpy().reshape(-1))
            correct += pred_.eq(target.view_as(pred_)).sum().item()

    test_loss /= len(test_loader)

    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct, 
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
        )
    )

    logging.info(metrics.classification_report(y_true = test_list,
        y_pred = pred_list,
        target_names  = gesture_config["train_labels_no_noise"]))

    cm = metrics.confusion_matrix(y_true = test_list,
        y_pred = pred_list)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    string = "\n"
    for row in cm:
        string += "  ".join(["{:.3f}".format(x) for x in row]) + "\n" 
    logging.info(string)

    return {"loss": test_loss, "true_label": test_list, "pred_label": pred_list}

def trainandsave_auto_encoderdecoder_classification(pytorch_encoder_model, pytorch_decoder_model, pytorch_fcclassification_model,
    train_loader, test_loader, logfile):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(device)

    encoder_model = pytorch_encoder_model.to(device)
    decoder_model = pytorch_decoder_model.to(device)
    fc_model = pytorch_fcclassification_model.to(device)

    early_stop_encoder = EarlyStopping(name = encoder_model.getname())
    early_stop_decoder = EarlyStopping(name = decoder_model.getname())
    early_stop_fc = EarlyStopping(name = fc_model.getname())

    criterion_reconstruction = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()
    lossfunction_reconstruction = nn.functional.mse_loss
    lossfunction_classification = nn.functional.cross_entropy

    lr = 0.0005
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer_encoder = optim.SGD(encoder_model.parameters(), lr=lr, momentum = 0.9)
    # optimizer_decoder = optim.SGD(decoder_model.parameters(), lr=lr, momentum = 0.9)

    optimizer_encoder = optim.Adam(encoder_model.parameters(), lr=lr)
    optimizer_decoder = optim.Adam(decoder_model.parameters(), lr=lr)
    optimizer_fc = optim.Adam(fc_model.parameters(), lr=lr)

    # train
    train_error = []
    test_error = []
    for epoch in range(200):
        print(epoch)
        if (epoch % 20 == 19):
            lr /= 1.2
            # optimizer_encoder = optim.SGD(encoder_model.parameters(), lr=lr, momentum = 0.9)
            # optimizer_decoder = optim.SGD(decoder_model.parameters(), lr=lr, momentum = 0.9)
            optimizer_encoder = optim.Adam(encoder_model.parameters(), lr=lr)
            optimizer_decoder = optim.Adam(decoder_model.parameters(), lr=lr)
            optimizer_fc = optim.Adam(fc_model.parameters(), lr=lr)
        
        train_results = train_reconstruction_three_models(encoder_model, decoder_model,fc_model,
            device, train_loader,
            criterion_reconstruction, criterion_classification,
            optimizer_encoder, optimizer_decoder, optimizer_fc, epoch)
        test_results = test_reconstruction_three_models(encoder_model, decoder_model,fc_model,
            device, test_loader,
            lossfunction_reconstruction, lossfunction_classification)
        train_error.append(train_results["loss"])
        test_error.append(test_results["loss"])
        with open(logfile.replace(".log", "_loss.txt"), "w") as f:
            f.write("\n".join([str(x) + "," + str(y) for x, y in zip(train_error, test_error)]))
        early_stop_encoder(test_results["loss"], encoder_model)
        early_stop_decoder(test_results["loss"], decoder_model)
        early_stop_fc(test_results["loss"], fc_model)

        if early_stop_decoder.early_stop:
            print("Early stopping")
            break

    with open("final_img_test.pkl", "wb") as f:
        pickle.dump(test_results,f)

    logging.info('Finished Training!')
    # save model
    encoder_model.load_state_dict(torch.load(early_stop_encoder.model_filename))
    encoder_model_name = encoder_model.getname()
    torch.save(encoder_model, encoder_model_name + '.pkl')
    torch.save(encoder_model.state_dict(), encoder_model_name + '_params.pkl')

    decoder_model.load_state_dict(torch.load(early_stop_decoder.model_filename))
    decoder_model_name = decoder_model.getname()
    torch.save(decoder_model, decoder_model_name + '.pkl')
    torch.save(decoder_model.state_dict(), decoder_model_name + '_params.pkl')

    fc_model.load_state_dict(torch.load(early_stop_fc.model_filename))
    fc_model_name = fc_model.getname()
    torch.save(fc_model, fc_model_name + '.pkl')
    torch.save(fc_model.state_dict(), fc_model_name + '_params.pkl')
