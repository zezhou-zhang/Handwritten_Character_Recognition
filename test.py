#! /bin/env python3
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split, cross_val_score

def load_pkl(fname):
	with open(fname,'rb') as f:
		return pickle.load(f)

def save_pkl(fname,obj):
	with open(fname,'wb') as f:
		pickle.dump(obj,f)

train_data = load_pkl('train_data.pkl')

labels = np.load('finalLabelsTrain.npy')
# Resize the image to make them all 48*48
def resize_data_image(data):
	if(len(data) !=  48):
		if(len(data)<48):
			if((50-len(data))%2 != 0):
				data = np.pad(data, [(((48-len(data))//2)+1, (48-len(data))//2), (0, 0)], mode='constant')
			else:
				data = np.pad(data, [((48-len(data))//2, (48-len(data))//2), (0, 0)], mode='constant')
		else:
			for i in range(len(data)):
				if(i >= 48):
					data = np.delete(data, 48, 0)
	if(len(data[0])!=48):
		if(len(data[0])<48):
			if((48-len(data[0]))%2 != 0):
				data = np.pad(data, [(0, 0), (((48-len(data[0]))//2)+1, (48-len(data[0]))//2)], mode='constant')
			else:
				data = np.pad(data, [(0, 0), (((48-len(data[0]))//2), (48-len(data[0]))//2)], mode='constant')
		else:
			for i in range(len(data[0])):
				if(i >= 48):
					data = np.delete(data, 48, 1)
	return data

resized_data = []
# Store all the images to resized_data and check if any data is not 48*48
for i in range(len(train_data)):
	resized_data.append(resize_data_image(train_data[i]))
	if (np.shape(resized_data[i]) != (48,48)):
		print("WRONG!")

for i in range(len(labels)):
    labels[i] = labels[i] - 1



'This part is used to extract a and b letter from dataset, if not going to use only a and b, please comment this area'
'''======================================================================================='''
#Extract the a and b dataset
a_b_dataset = []
a_b_labels = []
for i in range(len(labels)):
    if labels[i]== 0 or labels[i] == 1:
        a_b_dataset.append(resized_data[i])
        a_b_labels.append(labels[i])

#Convert the tensor type for pytorch
a_b_dataset = np.array(a_b_dataset)
a_b_dataset = torch.Tensor(a_b_dataset)
a_b_labels = np.array(a_b_labels)
a_b_labels = torch.Tensor(a_b_labels)
'''======================================================================================='''



resized_data = np.array(resized_data)
resized_data = torch.Tensor(resized_data)
labels = np.array(labels)
labels = torch.Tensor(labels)
resized_data = torch.unsqueeze(resized_data, dim=1)  # add one dimension--1


'This part is used to extract a and b letter from dataset, if not going to use only a and b, please comment this area'
'''======================================================================================='''
a_b_dataset = torch.unsqueeze(a_b_dataset, dim=1)  # add one dimension--1
a_b_test = []
for i in range(len(a_b_dataset)):
   a_b_test.append((a_b_dataset[i],a_b_labels[i]))
print("The length of a_b_test: " + str(len(a_b_test)))
a_b_test_loader = DataLoader(dataset = a_b_test, batch_size =len(a_b_test), shuffle = False)
'''======================================================================================='''



train_dataset = []
for i in range(len(resized_data)):
	train_dataset.append((resized_data[i],labels[i]))

train_loader, test_loader = train_test_split(train_dataset, test_size = .1)
test_loader = DataLoader(dataset=test_loader, batch_size=len(test_loader), shuffle=False)





class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(12 * 12 * 128, 1000)
        self.fc2 = nn.Linear(1000, 8)

# Forward

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet()

model.layer1.load_state_dict(torch.load('net_params_layer1.pkl'))
model.layer2.load_state_dict(torch.load('net_params_layer2.pkl'))
model.fc1.load_state_dict(torch.load('net_params_linear1.pkl'))
model.fc2.load_state_dict(torch.load('net_params_linear2.pkl'))

# Test the model
model.eval()
'''======================================================================================='''
'This part is used to extract a and b letter from dataset, if not going to use only a and b'
'Please change the a_b_test_loader to test_loader in line 148'
'''======================================================================================='''
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in a_b_test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.long() == labels.long()).sum().item()
        '''
        for i in range(len(labels)):
            if labels[i] == 0:
                print("The label of a is predicted as: " + str(int(predicted[i])+1) + '\n')
            if labels[i] == 1:
                print("The label of b is predicted as: " + str(int(predicted[i])+1) + '\n')
        '''
        predicted = list(predicted)
        labels = list(labels)
        for i in range(len(labels)):
            predicted[i] = int(predicted[i])+1
        for i in range(len(labels)):
            labels[i] = int(labels[i])+1
        print("The predicted labels are: "+ str(list(predicted)))
        print("The actual labels are:    "+ str(list(labels)))
    print("accuracy: "+ str(correct/total))


