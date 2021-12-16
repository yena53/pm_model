import torch
import torch.nn as nn
import joblib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):

  def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
    super(RNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid())

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state 설정하기.
    out, _ = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
    out = out.reshape(out.shape[0], -1) # many to many 전략
    out = self.fc(out)
    return out

input_size = 7
num_layers = 2
hidden_size = 64
sequence_length = 2

model = RNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)

PATH = 'model_호암동포함.pth'
model.load_state_dict(torch.load(PATH))
model.eval() # why we need this?

def transform(input_web):
    scaler = joblib.load('scaler.save')
    input_web=scaler.transform(input_web)
    input_web=torch.Tensor(input_web)
    return input_web

def get_prediction(input_tensor):
    input_tensor=input_tensor.reshape(-1, sequence_length, input_size).to(device)
    outputs=model(input_tensor)
    return outputs*1000