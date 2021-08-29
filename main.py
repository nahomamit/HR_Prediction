import csv
from time import sleep

import sklearn
from sklearn import utils
from torch.nn import BCELoss
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from numpy import vstack
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os.path


class HR_Moudle(nn.Module):
    def __init__(self, image_size):
        super(HR_Moudle, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)
        self.batchnorm1 = nn.BatchNorm1d(self.image_size, affine=False)
        self.batchnorm2 = nn.BatchNorm1d(64, affine=False)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.batchnorm1(x)
        x = F.relu(self.fc0(x))
        x = self.batchnorm2(x)
        x = F.relu(self.fc1(x))
        x = self.batchnorm2(x)
        out = F.relu(self.fc2(x))
        return torch.sigmoid(out)


Department = ['sales', 'accounting', 'hr', 'technical', 'support', 'management',
              'IT', 'product_mng', 'marketing', 'RandD']
Salary = ['low', 'medium', 'high']
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.05
model_filename = "HR_model"
model_test_x_file = "test_X.npy"
model_test_y_file = "test_Y.npy"
file_name = "HR_Employee_Data.xlsx"
sheet = "HR_Employee_Data"


def build_dataloader():
    df = pd.read_excel(io=file_name, sheet_name=sheet)
    df = df.to_numpy()
    df = np.array(df)
    df_x = list()
    df_y = list()

    for line in df:
        department = np.zeros(len(Department), dtype=int)
        salary = np.zeros(len(Salary), dtype=int)

        df_y.append(line[7])
        department[Department.index(line[9])] = 1
        salary[Salary.index(line[10])] = 1
        df_x.append(np.concatenate((line[1:6], line[8], department, salary), axis=None))

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.33333)
    # transform to torch tensor
    tensor_train_x = torch.Tensor(x_train)
    tensor_train_y = torch.Tensor(y_train)
    tensor_test_x = torch.Tensor(x_test)
    tensor_test_y = torch.Tensor(y_test)

    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)  # create your datset
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)  # create your datset

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # create your dataloader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)  # create your dataloader

    return train_loader, test_loader, x_test, y_test


# train the model
def train(train_dl, model):
    model.train()
    # define the optimization
    criterion = BCELoss()
    # enumerate epochs
    for epoch in range(EPOCHS):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            yhat = yhat.reshape(-1)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
    model.eval()
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# make a class prediction for one row of data
def predict(vector, model):
    model.eval()
    # convert row to data
    vector = np.array(vector)
    tensor_x = torch.from_numpy(vector)
    # data_set = TensorDataset(tensor_x)
    # data_loader = DataLoader(data_set, batch_size=1)
    input = tensor_x.unsqueeze(0)
    yhat = model(input)
    yhat = yhat.detach().numpy()
    yhat = yhat.round()

    return yhat


if __name__ == '__main__':
    global optimizer
    my_model = HR_Moudle(19)
    optimizer = Adam(my_model.parameters(), lr=LR)
    if os.path.isfile(model_filename) and os.path.isfile(model_test_x_file) and os.path.isfile(model_test_y_file):
        my_model.load_state_dict(torch.load(model_filename))
        x_test = np.load("test_X.npy", allow_pickle=True)
        y_test = np.load("test_Y.npy", allow_pickle=True)
    else:
        print("no model exist")
        train_loader, valid_loader, x_test, y_test = build_dataloader()
        train(train_loader, my_model)
        acc = evaluate_model(valid_loader, my_model)
        print('Accuracy: %.3f' % acc)
        sleep(1)
        if acc >= 0.90:
            torch.save(my_model.state_dict(), model_filename)
            np.save("test_X", x_test)
            np.save("test_Y", y_test)
    sum = 0
    x_test = np.array(x_test)
    x_test, y_test = utils.shuffle(x_test, y_test)
    left_worker = 0
    left_worker_right_predict = 0
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        x = x.astype('float32')
        y_hat = predict(x, my_model)
        if y == y_hat:
            sum += 1
        if y == 1:
            print(f"index {i}: predict: {y_hat}, target: {y}")
            left_worker += 1
            if y_hat == 1:
                left_worker_right_predict += 1
    print(f"total correct in {sum} cases that is {sum / 10}%")
    print(f"from {left_worker} that were left the company the model predict {left_worker_right_predict}\n"
          f"that is {left_worker_right_predict * 100 / left_worker}%")
