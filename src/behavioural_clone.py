### Use behavioural cloning with M-FOS state-action pairs as the 'expert'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data(file, batch_size=4096, episodes=2048, append_input=False):
    data = json.load(open(file))
    if append_input == True:
        state_data = np.zeros((batch_size*episodes, 7))
        action_data = np.zeros((batch_size*episodes, 7))
    else:
        state_data = np.zeros((batch_size*episodes, 5))
        action_data = np.zeros((batch_size*episodes, 5))

    cycle = 0
    for i in range(len(data[0::batch_size])):
        for j in range(batch_size):
            state_data[j+cycle][:] = data[j+cycle]['state_agent'+str(j)]
            action_data[j+cycle][:] = data[j+cycle]['action_agent'+str(j)]
        cycle += batch_size

    return torch.sigmoid(torch.nan_to_num(torch.Tensor(state_data))), torch.sigmoid(torch.nan_to_num(torch.Tensor(action_data)))


class NeuralNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BehaviouralCloning:
    def __init__(self, state_data, action_data):
        super(BehaviouralCloning, self).__init__()
        self.state_data = state_data
        self.action_data = action_data

    def run(self):
        # Hyperparameters
        state_dim = self.state_data.shape[1]
        action_dim = self.action_data.shape[1]
        learning_rate = 0.001
        num_epochs = 100

        state_train, state_val, state_test = torch.split(self.state_data, [int(0.8046875*len(self.state_data)), int(0.09765625*len(self.state_data)), int(0.09765625*len(self.state_data))])
        action_train, action_val, action_test = torch.split(self.action_data, [int(0.8046875*len(self.action_data)), int(0.09765625*len(self.action_data)), int(0.09765625*len(self.action_data))])

        # Convert your data to PyTorch tensors
        state_train_tensor = torch.Tensor(state_train)
        action_train_tensor = torch.Tensor(action_train)
        state_val_tensor = torch.Tensor(state_val)
        action_val_tensor = torch.Tensor(action_val)
        state_test_tensor = torch.Tensor(state_test)
        action_test_tensor = torch.Tensor(action_test)

        # Create a PyTorch DataLoader for training (optional but useful for batch processing)
        train_dataset = TensorDataset(state_train_tensor, action_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

        # Create the model
        model = NeuralNet(state_dim, action_dim)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            predictions = model(self.state_data) # Forward pass
            loss = criterion(predictions, self.action_data) # Compute the loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training loss for this epoch
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item()}")

            model.eval()
            with torch.no_grad():
                val_predictions = model(state_val_tensor)
                val_loss = criterion(val_predictions, action_val_tensor)
                print(f"Validation Loss: {val_loss.item()}")

            # Evaluation on the test set (for final evaluation)
            test_predictions = model(state_test_tensor)
            test_loss = criterion(test_predictions, action_test_tensor)
            print(f"Test Loss: {test_loss.item()}")

            print("action:", action_val_tensor)

        return action_val_tensor

