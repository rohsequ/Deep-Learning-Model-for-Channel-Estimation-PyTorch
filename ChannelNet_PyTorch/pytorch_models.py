import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from tqdm import tqdm
from datetime import datetime
import os

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y__%H_%M")
os.mkdir(f"saved_models\model_repo\{dt_string}")

class SRCNN(torch.nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()

        def init_weights(m):
            if type(m) == torch.nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)


        # L1 ImgIn shape=(?, 28, 28, 1)
        # # Conv -> (?, :, :, 64)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(p=1 - keep_prob)
        )
        self.layer1.apply(init_weights)
        # L2 ImgIn shape=(?, :, :, 64)
        # Conv      ->(?, :, :, 32)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(p=1 - keep_prob)
        )
        self.layer2.apply(init_weights)
        # L3 ImgIn shape=(?, :, :, 32)
        # Conv ->(?, :, :, 1)
        self.layer3 = torch.nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)
        torch.nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')
        self.layer3.bias.data.fill_(0.01)             

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out




class DnCNN(torch.nn.Module):

    def __init__(self):
        super(DnCNN, self).__init__()

        def init_weights(m):
            if type(m) == torch.nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)

        # L1 ImgIn shape=(?, 28, 28, 1)
        # Conv -> (?, :, :, 64)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(p=1 - keep_prob)
        )
        self.layer1.apply(init_weights)
        # L2 ImgIn shape=(?, :, :, 64)
        # Conv      ->(?, :, :, 64)
        # 18 layers, Conv2d + BN + ReLu
        layers = []
        for i in range(18):
            layers.append(torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)) 
            layers.append(torch.nn.BatchNorm2d(64, eps=1e-3))
            layers.append(torch.nn.ReLU())

        self.layer2 = torch.nn.Sequential(*layers)
        self.layer2.apply(init_weights)

        # L3 ImgIn shape=(?, :, :, 64)
        # Conv ->(?, :, :, 1)
        self.layer3 = torch.nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)
        torch.nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')
        self.layer3.bias.data.fill_(0.01) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        # input - noise. Because the model learns the noise parameters.
        out = torch.sub(x, out)

        return out

def train_SRCNN(train_size, test_size, dataset, n_epochs, model_name, device, load_from_checkpoint=None):

    print("Training the SRCNN model:")  
    print("##########################################")
    if model_name == "SRCNN":
        model = SRCNN()
    else:
        raise Exception("The Correct Model name WASNT given, Please check..")

    if load_from_checkpoint:
        if os.path.isfile(f"saved_models\{model_name}_checkpoint_latest.pt"):
            model.load_state_dict(torch.load(f"saved_models\{model_name}_checkpoint_latest.pt"))

    print(f"Model Loaded in Device: {device}")
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    running_loss=0
    total=0
    train_losses = []
    train_loss = 0
    after_train_loss = 0

    pbar = tqdm(range(n_epochs))

    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    for epoch in pbar:
        # Evaluation before Training
        # before_train_loss = 0
        # model.eval()
        # for batch_idx, data in enumerate(test_dataloader):
        #     x_test, y_test = data[0].to(device), data[1].to(device)
        #     y_pred = model(x_test)
        #     before_train = criterion(y_pred, y_test)
        #     before_train_loss += before_train.item()
        # print('Test loss before training: %.3f'% (before_train_loss / len(test_dataloader)))

        # Model going to train
        model.train()
        for batch_idx, data in enumerate(train_dataloader):
            
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs=model(inputs)

            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            pbar.set_description('Epoch: %d | Loss: %.3f| Avg Loss: %.3f | Eval Loss: %.3f'%(epoch, loss.item(), train_loss, after_train_loss/len(test_dataloader)))
        train_loss= running_loss/len(train_dataloader)
        running_loss = 0
        train_losses.append(train_loss)
        print(" ")
        # Evaluation After Training
        after_train_loss = 0
        model.eval()
        for data in test_dataloader:
            x_test, y_test = data[0].to(device), data[1].to(device)
            y_pred = model(x_test)
            after_train = criterion(y_pred, y_test)
            after_train_loss += after_train.item()
        # print('Test loss AFTER training: %.3f'% (after_train_loss / len(test_dataloader)))
        
        # Save the model
    print("##########################################")
    torch.save(model.state_dict(), f"saved_models\{model_name}_checkpoint_latest.pt")

    torch.save(model.state_dict(), f"saved_models\model_repo\{dt_string}\{model_name}_{dt_string}_checkpoint.pt")


def train_DnCNN(train_size, test_size, dataset, n_epochs, model_name, device, path_to_SRCNN, load_from_checkpoint=None):

    print("Training the DnCNN model:")  
    print("##########################################")
    if model_name == "DnCNN":
        model = DnCNN()
        srcnn_model = SRCNN()
        srcnn_model.load_state_dict(torch.load(path_to_SRCNN))
    else:
        raise Exception("The Correct Model name WASNT given, Please check..")

    if load_from_checkpoint:
        if os.path.isfile(f"saved_models\{model_name}_checkpoint_latest.pt"):
            model.load_state_dict(torch.load(f"saved_models\{model_name}_checkpoint_latest.pt"))
    print(f"Model Loaded in Device: {device}")
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    running_loss=0
    total=0
    train_losses = []
    train_loss = 0
    after_train_loss = 0
    pbar = tqdm(range(n_epochs))

    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    for epoch in pbar:
        # Evaluation before Training
        # before_train_loss = 0
        # model.eval()
        # for batch_idx, data in enumerate(test_dataloader):
        #     x_test, y_test = data[0].to(device), data[1].to(device)
        #     x_test_dash = srcnn_model(x_test)
        #     y_pred = model(x_test_dash.copy())
        #     before_train = criterion(y_pred, y_test)
        #     before_train_loss += before_train.item()
        # print('Test loss before training: %.3f'% (before_train_loss / len(test_dataloader)))

        # Model going to train
        model.train()
        for batch_idx, data in enumerate(train_dataloader):
            
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs=model(inputs)

            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            pbar.set_description('Epoch: %d | Loss: %.3f| Avg Loss: %.3f | Eval Loss: %.3f'%(epoch, loss.item(), train_loss, after_train_loss/len(test_dataloader)))
        train_loss= running_loss/len(train_dataloader)
        running_loss = 0
        train_losses.append(train_loss)
        print(" ")
        # Evaluation After Training
        after_train_loss = 0
        model.eval()
        for data in test_dataloader:
            x_test, y_test = data[0].to(device), data[1].to(device)
            y_pred = model(x_test)
            after_train = criterion(y_pred, y_test)
            after_train_loss += after_train.item()
        # print('Test loss AFTER training: %.3f'% (after_train_loss / len(test_dataloader)))
        
        # # Save the model
    print("##########################################")

    torch.save(model.state_dict(), f"saved_models\model_repo\{dt_string}\{model_name}_{dt_string}_checkpoint.pt")

    torch.save(model.state_dict(), f"saved_models\{model_name}_checkpoint_latest.pt")

        