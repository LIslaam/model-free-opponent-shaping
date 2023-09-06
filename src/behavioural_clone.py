### Use behavioural cloning with M-FOS state-action pairs as the 'expert'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data(file, episodes, batch_size, append_input=False):
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

    def __enter__(self):
        return self

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BehaviouralCloning:
    def __init__(self, state_data, action_data, batch_size=4096):
        super(BehaviouralCloning, self).__init__()
        self.state_data = state_data.to(device)
        self.action_data = action_data.to(device)
        self.batch_size = batch_size
        self.state_dim = self.state_data.shape[1]
        self.action_dim = self.action_data.shape[1]

        # Create the model
        self.model = NeuralNet(self.state_dim, self.action_dim).to(device)

    def run(self):
        # Hyperparameters
        model = self.model
        learning_rate = 0.001
        num_epochs = 100

        state_train, state_val, state_test = torch.split(self.state_data, [int(0.8046875*len(self.state_data)), int(0.09765625*len(self.state_data)), int(0.09765625*len(self.state_data))])
        action_train, action_val, action_test = torch.split(self.action_data, [int(0.8046875*len(self.action_data)), int(0.09765625*len(self.action_data)), int(0.09765625*len(self.action_data))])

        # Convert your data to PyTorch tensors
        state_train_tensor = state_train.clone().detach().requires_grad_(True)
        action_train_tensor = action_train.clone().detach().requires_grad_(True)
        state_val_tensor = state_val.clone().detach().requires_grad_(True)
        action_val_tensor = action_val.clone().detach().requires_grad_(True)
        state_test_tensor = state_test.clone().detach().requires_grad_(True)
        action_test_tensor = action_test.clone().detach().requires_grad_(True)

        # Create a PyTorch DataLoader for training (optional but useful for batch processing)
        train_dataset = TensorDataset(state_train_tensor, action_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

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

            model.train()

        model.eval()
        with torch.no_grad():
            # Evaluation on the test set (for final evaluation)
            test_predictions = model(state_test_tensor)
            test_loss = criterion(test_predictions, action_test_tensor)
            print(f"Test Loss: {test_loss.item()}")

    
    def evaluate(self, state):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(state)

        return predictions

# A TEST
#state_data, action_data = get_data("runs/DATA_COLLECTION_mfos_ppo_input_ipd_randopp_nl/state_action/out_512.json", 512, 4096)
#clone = BehaviouralCloning(state_data, action_data)
#clone.run()
#print (clone.evaluate(torch.tensor([0.1231,0.5,0,0.5,0.5]).to(device)))
#print (clone.evaluate(torch.tensor([[0.5,0.5,0.5,0.5,0.5], [0.1231,0.5,0,0.5,0.5]]).to(device)))