import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from imageio import imread
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#create train dataset
def create_train_dataset():
    t3 = torch.zeros((24,2,800,800))
    for i in range(1,25):
        test_image = imread('BIA_final/IM'+str(i)+'.png') #read in images from folder
        if test_image.shape != (1035,1280,3):
            test_image = test_image[175:975,500:1300,:] #confirm 800x800 shape
        else:
            test_image = test_image[175:975,275:1075,:] #confirm 800x800 shape
        img = test_image[:,:,0]
        print(img.shape)
        new_img = np.zeros([img.shape[0],img.shape[1]])

        threshold = 120 
        #threshold image to highlight noise
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if img[j,k] < threshold:
                    new_img[j,k] = 0 
                else:
                    new_img[j,k] = 1 
        #make train set one layer with input, one layer with thresholded image
        tensor = torch.tensor([img,new_img]) 
        t3[i-1,:,:,:] = tensor
        
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand
 
#Model
model = UNET(1,1) #call model with in and output channels

#Loss
loss_fn = nn.CrossEntropyLoss() #cross entropy loss with photo input

#Optimizer
lr_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

#Dataset
X_train,X_test,Y_train,Y_test = train_test_split(t3[:,0,:,:],t3[:,1,:,:])
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

class Dataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X,dtype=torch.float)
        self.y = torch.tensor(y,dtype=torch.float)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
 
#forms dataset
train_dataset = Dataset(X_train,Y_train)
test_dataset = Dataset(X_test,Y_test)

#create dataloaders
train_dataloader = DataLoader(train_dataset,batch_size=2,shuffle=True,num_workers=1)
test_dataloader = DataLoader(test_dataset,batch_size=2,shuffle=True,num_workers=1)

#Training Functions
def train(model, train_dl,loss_fn, optimizer, acc_fn, epochs):
    train_loss = []
    best_acc = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        model.train(True)
#         for phase in ['train', 'valid']:
#             if phase == 'train':
#                 model.train(True)  # Set trainind mode = true
#                 dataloader = train_dl
#             else:
#                 model.train(False)  # Set model to evaluate mode
#                 dataloader = valid_dl
        running_loss = 0.0
        running_acc = 0.0
        step = 0
        dataloader = train_dataloader
        # iterate over data
        for x, y in dataloader:
            step += 1
                # forward pass
            #if phase == 'train':
                # zero the gradients
            optimizer.zero_grad()
            outputs = model(x.unsqueeze(dim=1))
            loss = loss_fn(outputs.squeeze(), y)
                # the backward pass frees the graph memory, so there is no 
                # need for torch.no_grad in this training pass
            loss.backward()
            optimizer.step()
            # scheduler.step()
            #else:
                #with torch.no_grad():
                    #outputs = model(x)
                    #loss = loss_fn(outputs, y.long())
            acc = acc_fn(outputs, y)
            running_acc  += acc*dataloader.batch_size
            running_loss += loss*dataloader.batch_size 

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = 1 - running_acc / len(dataloader.dataset)

        print('Loss: {:.4f} Acc: {}'.format(epoch_loss, epoch_acc))

        train_loss.append(epoch_loss) #if phase=='train' else valid_loss.append(epoch_loss)
    return train_loss

#Accuracy metric for Model
def acc_metric(predb, yb):
    acc = 0
    z=expit(predb.squeeze().detach().numpy())
    z2 = np.zeros([z.shape[0],z.shape[1],z.shape[2]])
    for i in range(z.shape[0]):
        threshold = .5
        for j in range(z.shape[1]):
            for k in range(z.shape[2]):
                if z[i,j,k] <= threshold:
                    z2[i,j,k] = 0
                else:
                    z2[i,j,k] = 1
        neg = 0
        diff = yb[i]-z2[i]
        for j in range(diff.shape[0]):
            for k in range(diff.shape[1]):
                if diff[i,j] == -1.0 or diff[i,j] == 1.0:
                    neg += 1
        acc += neg/(diff.shape[0]**2)
    return acc

 
#Plot training/validation loss  
plt.figure(figsize=(10,8))
train_loss_2 = []
for i in train_loss:
    train_loss_2.append(i.detach().numpy())
plt.plot(train_loss_2, label='Train loss')
plt.legend()
