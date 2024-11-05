import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Hyperparameters
input_size = 128  # Number of unique characters (V in the approach)
hidden_size = 512
num_layers = 2
num_epochs = 5
learning_rate = 0.001
sequence_length = 100  # Number of characters per sequence
batch_size = 64

# Load and preprocess data
def load_codeforces_data(folder_path):
    code_data = ""
    chars = set()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.java', '.py', '.cpp')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    code_data += code
                    chars.update(code)

    chars = sorted(list(chars))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    encoded_data = [char_to_idx[c] for c in code_data if c in char_to_idx]

    return encoded_data, char_to_idx, idx_to_char

encoded_data, char_to_idx, idx_to_char = load_codeforces_data("C:\\Users\\kulam\\OneDrive\\Documents\\Codeforces_Dataset\\New_Db")
vocab_size = len(char_to_idx)

# Split data into training and testing sets
train_data, test_data = train_test_split(encoded_data, test_size=0.2, random_state=42)

# RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(-1, self.hidden_size))
        return out, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# Data loader function
def create_dataloader(data, sequence_length, batch_size):
    sequences = []
    for i in range(0, len(data) - sequence_length, sequence_length):
        seq = torch.tensor(data[i:i + sequence_length], dtype=torch.long)
        sequences.append(seq)
    return torch.utils.data.DataLoader(sequences, batch_size=batch_size, shuffle=True)

# Create dataloaders
train_loader = create_dataloader(train_data, sequence_length, batch_size)
test_loader = create_dataloader(test_data, sequence_length, batch_size)

# Train and evaluate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharRNN(vocab_size, hidden_size, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_correct = 0
    total_train_predictions = 0
    
    for i, batch in enumerate(train_loader):
        batch_size = batch.size(0)
        hidden = model.init_hidden(batch_size, device)

        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:].reshape(-1)

        hidden = tuple([h.detach() for h in hidden])
        outputs, hidden = model(inputs, hidden)
        outputs = outputs.view(batch_size * (sequence_length - 1), vocab_size)

        # Calculate loss and update model
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, dim=1)
        total_train_correct += (predicted == targets).sum().item()
        total_train_predictions += targets.size(0)

    train_accuracy = total_train_correct / total_train_predictions * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Evaluation phase
    model.eval()
    total_test_correct = 0
    total_test_predictions = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch_size = batch.size(0)
            hidden = model.init_hidden(batch_size, device)

            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:].reshape(-1)
            outputs, hidden = model(inputs, hidden)
            outputs = outputs.view(batch_size * (sequence_length - 1), vocab_size)

            # Calculate test accuracy
            _, predicted = torch.max(outputs, dim=1)
            total_test_correct += (predicted == targets).sum().item()
            total_test_predictions += targets.size(0)

    test_accuracy = total_test_correct / total_test_predictions * 100
    print(f"Epoch [{epoch+1}/{num_epochs}] Test Accuracy: {test_accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "C:\\Projects\\Major_ML\\Saved_Path\\char_rnn_model.pth")
