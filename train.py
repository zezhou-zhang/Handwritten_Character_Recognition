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

for i in range(len(train_data)):
	resized_data.append(resize_data_image(train_data[i]))
	if (np.shape(resized_data[i]) != (48,48)):
		print("WRONG!")

#Super unclear image index
delete_data_list = [980,981,982,983,984,985,985,987,988,989,
                    941,942,943,930,931,933,1060,1054,1609,2140,3285,3330,
                    3840,3850,3851,3852,3863,3856,3857,3858,3859,3860,3862,
                    3864,3865,3866,3870,3871,3872,3873,3875,3876,3877,3878,3879,
                    3883,3884,3885,3889,3890,3891,3894,3895]

#Delete those unclear image from dataset and corresponding labels
i=0 #loop counter
length = len(delete_data_list)  #list length 
for i in range(length):
    resized_data.pop(delete_data_list[i])
    labels = list(labels)
    labels.pop(delete_data_list[i])
    if i + 1 < len(delete_data_list):
        delete_data_list[i+1] = delete_data_list[i+1] -1 
# print list after removing given element
print(len(resized_data))
print(len(labels))

for i in range(len(labels)):
    labels[i] = labels[i] - 1

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
resized_data = np.array(resized_data)
resized_data = torch.Tensor(resized_data)
labels = np.array(labels)
labels = torch.Tensor(labels)
'''
for k in range(60,64):
    plt.figure(k)
    for i in range(k*100, (k+1)*100):
        plt.subplot(10,10,i-k*100+1)
        plt.imshow(resized_data[i],cmap = 'Greys')
    plt.show()
'''

resized_data = torch.unsqueeze(resized_data, dim=1)  # add one dimension--1
a_b_dataset = torch.unsqueeze(a_b_dataset, dim=1)  # add one dimension--1
print(resized_data.shape)



train_dataset = []
for i in range(len(resized_data)):
	train_dataset.append((resized_data[i],labels[i]))

a_b_test = []
for i in range(len(a_b_dataset)):
   a_b_test.append((a_b_dataset[i],a_b_labels[i]))

print("The length of a_b_test: " + str(len(a_b_test)))

train_loader, test_loader = train_test_split(train_dataset, test_size = .1)

train_loader = DataLoader(dataset=train_loader, batch_size=572, shuffle=True)
#  test_loader = DataLoader(dataset=test_loader, batch_size=1, shuffle=False)
a_b_test_loader = DataLoader(dataset = a_b_test, batch_size =len(a_b_test), shuffle = False)


num_epochs = 25
num_classes = 8
learning_rate = 0.0005



# Model
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




#Train the model
model = ConvNet()


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train loop
# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs): #epoch
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward
        outputs = model(images)
        loss = criterion(outputs, labels.long())
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted.long() == labels.long()).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

# Test the model
model.eval()
torch.save(model.layer1.state_dict(), 'net_params_layer1.pkl')
torch.save(model.layer2.state_dict(), 'net_params_layer2.pkl')
torch.save(model.fc1.state_dict(), 'net_params_linear1.pkl')
torch.save(model.fc2.state_dict(), 'net_params_linear2.pkl')
'''
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.long() == labels.long()).sum().item()
    print("accuracy: "+ str(correct/total))
'''
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in a_b_test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.long() == labels.long()).sum().item()
    print("accuracy: "+ str(correct/total))



