import torch
from torch import nn


class Net(nn.Module):
    '''
    Neural network class with 3 input nodes, a hidden layer of size 3 and a single output
    '''

    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))



def grad_descent(n, net, optimizer, criterion, X_train, y_train, X_test, y_test): 
    '''
    Function that takes n steps of gradient descent
    '''

    list_train_loss = []
    list_test_loss = []

    for epoch in range(n):
    
        # Predicted outputs
        y_pred = net(X_train)
        y_pred = torch.squeeze(y_pred)

        # Loss with respect to the selected criterion (BCE in our case)
        train_loss = criterion(y_pred, y_train)

        # Keep every 10th loss
        if epoch % 10 == 0:

            y_test_pred = net(X_test)
            y_test_pred = torch.squeeze(y_test_pred)


            list_train_loss.append(train_loss)
            
            y_pred = net(X_test)
            
            y_pred = torch.squeeze(y_pred)
            test_loss = criterion(y_pred, y_test)
            
            list_test_loss.append(test_loss)

        optimizer.zero_grad()
        
        train_loss.backward()
        
        optimizer.step()

    

    return list_train_loss, list_test_loss
