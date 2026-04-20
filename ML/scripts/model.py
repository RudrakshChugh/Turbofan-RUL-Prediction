import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, alpha=1.0):
        return GradientReversalFn.apply(x, alpha)


class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2,
                 bidirectional=True):
        super(BaselineLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        # Bidirectional doubles the output dimension
        fc_input = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(fc_input, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :] 
        
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        rul_pred = self.out(x)
        return rul_pred

class AdvancedCNNLSTM(nn.Module):
    def __init__(self, input_size, seq_len, num_domains=6, dropout_rate=0.4):
        super(AdvancedCNNLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout_rate
        )
        
        self.rul_fc1 = nn.Linear(64, 32)
        self.rul_drop1 = nn.Dropout(p=dropout_rate)
        self.rul_fc2 = nn.Linear(32, 16)
        self.rul_drop2 = nn.Dropout(p=dropout_rate)
        self.rul_out = nn.Linear(16, 1)
        
        self.grl = GradientReversalLayer()
        self.dom_fc1 = nn.Linear(64, 32)
        self.dom_drop1 = nn.Dropout(p=dropout_rate)
        self.dom_out = nn.Linear(32, num_domains)

    def forward(self, x, alpha=1.0):
        x_cnn = x.permute(0, 2, 1)
        
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.relu(x_cnn)
        
        x_lstm_in = x_cnn.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x_lstm_in)
        global_features = lstm_out[:, -1, :]
        
        rul = self.relu(self.rul_fc1(global_features))
        rul = self.rul_drop1(rul)
        rul = self.relu(self.rul_fc2(rul))
        rul = self.rul_drop2(rul)
        rul_pred = self.rul_out(rul)
        
        dom_feats = self.grl(global_features, alpha)
        dom = self.relu(self.dom_fc1(dom_feats))
        dom = self.dom_drop1(dom)
        dom_pred = self.dom_out(dom)
        
        return rul_pred, dom_pred
