### Use behavioural cloning with M-FOS state-action pairs as the 'expert'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

METHOD = 'supervised' # choose supervised or RNN methods

def inv_sigmoid(x):
    return -torch.log((1 / (x + 1e-8)) - 1)


if METHOD == 'supervised':
    def get_data(file, episodes, batch_size, append_input=False):
        data = json.load(open(file))
        if append_input == True:
            state_data = np.zeros((batch_size*episodes, 7))
            action_data = np.zeros((batch_size*episodes, 5))
        else:
            state_data = np.zeros((batch_size*episodes, 5))
            action_data = np.zeros((batch_size*episodes, 5))

        cycle = 0
        for i in range(len(data[0::batch_size])):
            for j in range(batch_size):
                state_data[j+cycle][:] = data[j+cycle]['state_agent'+str(j)]
                action_data[j+cycle][:] = data[j+cycle]['action_agent'+str(j)]
            cycle += batch_size

        return (torch.nan_to_num(torch.Tensor(state_data))), (torch.nan_to_num(torch.Tensor(action_data)))


    class NeuralNet(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(NeuralNet, self).__init__()
            #self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Linear(128, 128),
                nn.Dropout(p=0.75),
                nn.Linear(128, action_dim), # Matching states to actions
            )

            # Apply He initialization to the weights of the linear layers
            for layer in self.linear_relu_stack:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)

        def forward(self, x):
            #x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            #pred_prob = nn.Softmax(dim=0)(logits)
            return logits


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

            # Shuffle data in place!
            combined_lists = list(zip(self.state_data, self.action_data))
            random.shuffle(combined_lists)
            shuffled_state, shuffled_action = zip(*combined_lists)

            state_val, state_test, state_train = torch.split(self.state_data, [int(0.09765625*len(shuffled_state)), int(0.09765625*len(shuffled_state)), int(0.8046875*len(shuffled_state))])
            action_val, action_test, action_train = torch.split(self.action_data, [int(0.09765625*len(shuffled_action)), int(0.09765625*len(shuffled_action)), int(0.8046875*len(shuffled_action))])

            # Convert your data to PyTorch tensors
            state_train_tensor = state_train.clone().detach().requires_grad_(True)
            action_train_tensor = action_train.clone().detach().requires_grad_(True)
            state_val_tensor = state_val.clone().detach().requires_grad_(True)
            action_val_tensor = action_val.clone().detach().requires_grad_(True)
            state_test_tensor = state_test.clone().detach().requires_grad_(True)
            action_test_tensor = action_test.clone().detach().requires_grad_(True)

            # Normalise features:
            state_train_tensor = F.normalize(state_train_tensor, p=2, dim=0)
            state_val_tensor = F.normalize(state_val_tensor, p=2, dim=0)
            state_test_tensor = F.normalize(state_test_tensor, p=2, dim=0)

            # Define regularization strength (lambda)
            l1_lambda = 0.001
            l2_lambda = 0.001

            # Define loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1) # Adding learning rate annealing

            # Training loop
            for epoch in range(num_epochs):
                predictions = model(self.state_data) # Forward pass
                loss = criterion(predictions, self.action_data) # Compute the loss

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                # Apply L1 regularization to specific layers
                for layer in model.children():
                    if isinstance(layer, nn.Linear):
                        layer.weight.register_hook(lambda grad: l1_lambda * torch.sign(grad))
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
                state =  F.normalize(state, p=2, dim=0)
                predictions = self.model(state)

            return predictions

    # A TEST
    #state_data, action_data = get_data("runs/DATA_COLLECTION_mfos_ppo_input_ipd_randopp_nl/state_action/out_2048.json", 2048, 4096, True)
    #clone = BehaviouralCloning(state_data, action_data)
    #clone.run()
    #print (clone.evaluate((torch.tensor([0.5175, 0.7180, 0.5061, 0.5206, 0.5054, 0.0169, 0.0235])).to(device)))
    #print (clone.evaluate((torch.tensor([0.06391, 0.06333, 0.06272, 0.07026, 0.05948])).to(device)))

elif METHOD == 'RNN':
    def get_data(file, episodes, batch_size, append_input=False):
        data = json.load(open(file))
        if append_input == True:
            state_data = np.zeros((batch_size, episodes, 7))
            action_data = np.zeros((batch_size, episodes, 5)) # The final reward after training
        else:
            state_data = np.zeros((batch_size, episodes, 5))
            action_data = np.zeros((batch_size, episodes, 5)) # The final reward after training

        cycle = 0
        for i in range(len(data[0::batch_size])):
            for j in range(batch_size):
                state_data[j][i][:] = data[j+cycle]['state_agent'+str(j)]
                action_data[j][i][:] = data[j+cycle]['action_agent'+str(j)]
            cycle += batch_size

        #print((torch.nan_to_num(torch.Tensor(state_data)))[0][:2])
        #print((torch.nan_to_num(torch.Tensor(action_data)))[0][:2])

        return (torch.nan_to_num(torch.Tensor(state_data))).to(device), (torch.nan_to_num(torch.Tensor(action_data))).to(device)


    # Define the RNN-based model
    class RNN(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.rnn = nn.RNN(state_dim, hidden_size, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_size, action_dim)

        def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # num_layers, batch_size, hidden_size

            # Forward pass through the RNN
            out, _ = self.rnn(x, h0)

            # Predict actions from RNN output
            out = self.fc(out)
            action_probs = torch.softmax(out, dim=-1)
            return action_probs


    class BehaviouralCloning:
        def __init__(self, state_data, action_data, batch_size=4096, num_episodes=2048):
            super(BehaviouralCloning, self).__init__()
            self.state_data = state_data.to(device)
            self.action_data = action_data.to(device)
            self.batch_size = batch_size
            self.num_episodes = num_episodes
            self.state_dim = self.state_data.shape[-1]
            self.action_dim = self.action_data.shape[-1]
            self.hidden_size = 64  # RNN hidden state dimension

            # Create the model
            self.model = RNN(self.state_dim, self.action_dim, self.hidden_size).to(device)

        def run(self):
            # Hyperparameters
            learning_rate = 0.001
            num_epochs = 1000

            # Create the RNN model
            model = self.model

            # Define loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(self.state_data)
                #outputs = outputs.view(-1, self.action_dim)  # Flatten for the loss function
                #actions_flat = self.action_data.view(-1)

                loss = criterion(outputs, self.action_data)
                loss.backward()
                optimizer.step()

                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        def evaluate(self, state):
            while len(state.size()) < 3:
                state = torch.unsqueeze(state, 0)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(state)
                predictions = torch.squeeze(predictions)
            return predictions
        
    # A TEST
    #state_data, action_data = get_data("runs/DATA_COLLECTION_mfos_ppo_input_ipd_randopp_nl/state_action/out_512.json", 512, 4096, True)
    #clone = BehaviouralCloning(state_data, action_data)
    #clone.run()
    #print (clone.evaluate((torch.tensor(([[0.5175, 0.7180, 0.5061, 0.5206, 0.5054, 0.0169, 0.0235]]))).to(device)))
    #print (clone.evaluate((torch.tensor([0.06391, 0.06333, 0.06272, 0.07026, 0.05948])).to(device)))

else:
    raise NotImplementedError