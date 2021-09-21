'''Collection of various DeepLearning models for DRL'''
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQnet(nn.Module):
    '''
    Simple MLP for Deep-Q-Learning applications.
    '''
    def __init__(self, n_actions, lr, input_dims, fc1_dims, fc2_dims):
        super().__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value  = self.fc3(x)

        return value

