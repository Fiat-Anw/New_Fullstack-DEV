import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt # Visualization
import matplotlib.dates as mdates # Formatting dates
import seaborn as sns # Visualization
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("sensor_data.csv")
df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M')
df = df.rename(columns={'time': 'Date'})
df.set_index('Date', inplace=True)

# Setting 80 percent data for training
split_idx = math.ceil(len(df) * .8)
split_idx

# Splitting the dataset
train_data = df[:split_idx]
test_data = df[split_idx:]

# Select only temperature column
train_data = train_data.iloc[:, 2:3]
test_data = test_data.iloc[:, 2:3]

dataset_train = train_data.temp_bmp280.values
# Reshaping 1D to 2D array
dataset_train = np.reshape(dataset_train, (-1,1))

dataset_test = test_data.temp_bmp280.values
# Reshaping 1D to 2D array
dataset_test = np.reshape(dataset_test, (-1,1))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(dataset_train)
scaled_test = scaler.fit_transform(dataset_test)

sequence_length = 50 # Number of time steps to look back
X_train, y_train = [], []

for i in range(len(scaled_train) - sequence_length):
	X_train.append(scaled_train[i:i+sequence_length])
	y_train.append(scaled_train[i+1:i+sequence_length+1])

X_train, y_train = np.array(X_train), np.array(y_train)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

sequence_length = 30 # Number of time steps to look back
X_test, y_test = [], []

for i in range(len(scaled_test) - sequence_length):
	X_test.append(scaled_test[i:i+sequence_length])
	y_test.append(scaled_test[i+1:i+sequence_length+1])

X_test, y_test = np.array(X_test), np.array(y_test)

# Convert data to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

batch_size = 16

# Create DataLoader for train dataset
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create DataLoader for test dataset
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super(LSTMModel, self).__init__() #initializes the parent class nn.Module
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, 1)

	def forward(self, x):
		out, _ = self.lstm(x)
		out = self.linear(out)
		return out

input_size = 1
num_layers = 2
hidden_size = 64
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers).to(device)

loss_fn = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(model)

from tqdm.auto import tqdm

num_epochs = 5
train_hist =[]
test_hist =[]
# Training loop
for epoch in tqdm(range(num_epochs)):
	total_loss = 0.0

	# Training
	model.train()
	for batch_X, batch_y in train_loader:
		batch_X, batch_y = batch_X.to(device), batch_y.to(device)
		predictions = model(batch_X)

		loss = loss_fn(predictions, batch_y)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		total_loss += loss.item()

	# Calculate average training loss and accuracy
	average_loss = total_loss / len(train_loader)
	train_hist.append(average_loss)

	# Validation on test data
	model.eval()
	with torch.no_grad():
		total_test_loss = 0.0

		for batch_X_test, batch_y_test in test_loader:
			batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
			predictions_test = model(batch_X_test)
			test_loss = loss_fn(predictions_test, batch_y_test)

			total_test_loss += test_loss.item()

		# Calculate average test loss and accuracy
		average_test_loss = total_test_loss / len(test_loader)
		test_hist.append(average_test_loss)
	if (epoch + 1) % 10 == 0:
		print(f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')
  
x = np.linspace(1,num_epochs,num_epochs)
plt.plot(x,train_hist,scalex=True, label="Training loss")
plt.plot(x, test_hist, label="Test loss")
plt.legend()
plt.show()

