import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# generate synthetic data (x) 
train_test = np.linspace(0, 10, 150)

# convert numpy arrays to PyTorch tensors
x_train_test = torch.from_numpy(train_test).float().unsqueeze(1)

# increase st dev of noise based on p, parametrizing covariate shift
def generate_shift_noise(size, p):
    noise = np.random.normal(0, 0.2 + p/2.5, size)
    return torch.from_numpy(noise).float().unsqueeze(1)

p_train = 0.2    
size = len(x_train_test)
y_train = 0.1 * x_train_test + torch.sin(x_train_test) + generate_shift_noise(size, p_train) #

# basic dataset
class SampleDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]    
    
# create dataset and dataloader
train_dataset = SampleDataset(x_train_test, y_train)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# NN Model
model = nn.Sequential(
    nn.Linear(1, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

# MSE loss function, SGD optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# training loop
n_epochs = 40
for epoch in range(n_epochs):
    for Xbatch, ybatch in train_loader:
        # forward step, loss
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        
        # zero gradient
        optimizer.zero_grad()
        
	# backward propagation
        loss.backward()
        optimizer.step()
        
        print(f"Finished epoch {epoch}, latest loss {loss.item()}", end="\r", flush=True)
