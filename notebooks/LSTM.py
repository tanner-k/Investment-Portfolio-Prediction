import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


# Function to split pandas dataframe to PyTorch tensors
def splitting_to_tensor(df, future_pred=126, test_size=3): 
    """
    future_pred: Number of time steps into the future to predict.
    test_size: Number of portfolios in the test group.
    """

    transformed_df = df.T.to_numpy()

    train_input = torch.from_numpy(transformed_df[test_size:, :-future_pred])
    train_target = torch.from_numpy(transformed_df[test_size:, future_pred:])
    test_input = torch.from_numpy(transformed_df[:test_size, :-future_pred])
    test_target = torch.from_numpy(transformed_df[:test_size, future_pred:])

    return train_input, train_target, test_input, test_target

# Function to split the dataframe to a train and a test set. 
# However for LSTM the test set will not be used as the entire portfolio needs to be used to predict the future values. 
def splitting_tr_te(df, size=0.95):
    
    split_point = int(df.shape[0]*size)

    train = df.iloc[:split_point]
    test = df.iloc[split_point:]
    return train, test

# The LSTM model with 2 lstm cells
class LSTM(nn.Module):
    def __init__(self, n_hidden=128):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        """
        x: tensor of portfolios. (shape: portfolios, time steps)
        future: number of time steps to predict into the future.

        output: future predictions for each portfolio. (shape: portfolios, time steps + future)
        """

        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        for input_t in x.split(1, dim=1):
            # input_t shape: portfolios, 1 time step
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t, c_t))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs

model = LSTM().float()

# Function to train the LSTM model using LBFGS optimizer
def train_LSTM(train_input, train_target, test_input, test_target, epochs=10, 
               model=model, criterion=nn.MSELoss(), optimizer=optim.LBFGS(model.parameters(), lr=0.6, max_iter=3)):

    for epoch in range(epochs):
        print("Step:", epoch)

        def closure():
            optimizer.zero_grad()
            out = model(train_input.float())
            loss = criterion(out, train_target.float())
            print("Loss:", loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

        with torch.no_grad():
            pred = model(test_input.float())
            loss = criterion(pred, test_target.float())
            print("Test Loss:", loss.item())
    y = pred.detach().numpy()

    return y, loss
