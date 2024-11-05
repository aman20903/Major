import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os

# RNN model definition
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

def create_sequences(encoded_data, sequence_length):
    sequences = []
    for i in range(0, len(encoded_data) - sequence_length, sequence_length):
        seq = encoded_data[i:i + sequence_length]
        sequences.append(seq)
    return sequences

def create_dataloader(sequences, batch_size, shuffle=True):
    tensor_sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    return torch.utils.data.DataLoader(tensor_sequences, batch_size=batch_size, shuffle=shuffle)

def evaluate_model(model, dataloader, criterion, device, sequence_length):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch.size(0)
            hidden = model.init_hidden(batch_size, device)
            
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:].reshape(-1)
            
            outputs, _ = model(inputs, hidden)
            outputs = outputs.view(batch_size * (sequence_length - 1), -1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == targets).sum().item()
            total_predictions += targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_predictions) * 100
    return avg_loss, accuracy

def main():
    # Hyperparameters
    input_size = 128
    hidden_size = 512
    num_layers = 2
    num_epochs = 5
    learning_rate = 0.001
    sequence_length = 100
    batch_size = 64
    test_size = 0.2

    # Load data
    encoded_data, char_to_idx, idx_to_char = load_codeforces_data("C:\\Projects\\Major_ML\\New_Db")
    vocab_size = len(char_to_idx)
    
    # Create sequences
    sequences = create_sequences(encoded_data, sequence_length)
    
    # Split into train and test sets
    train_sequences, test_sequences = train_test_split(sequences, test_size=test_size, random_state=42)
    
    # Create dataloaders
    train_dataloader = create_dataloader(train_sequences, batch_size)
    test_dataloader = create_dataloader(test_sequences, batch_size, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharRNN(vocab_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_test_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_predictions = 0
        
        for i, batch in enumerate(train_dataloader):
            batch_size = batch.size(0)
            hidden = model.init_hidden(batch_size, device)
            
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:].reshape(-1)
            
            hidden = tuple([h.detach() for h in hidden])
            outputs, hidden = model(inputs, hidden)
            outputs = outputs.view(batch_size * (sequence_length - 1), vocab_size)
            
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == targets).sum().item()
            total_predictions += targets.size(0)
            
            if i % 100 == 0:
                train_accuracy = total_correct / total_predictions * 100
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], "
                      f"Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device, sequence_length)
        print(f"Epoch [{epoch+1}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Save best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_accuracy,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char
            }, "C:\\Projects\\Major_ML\\Saved_Path\\best_char_rnn_model.pth")
            print(f"Saved new best model with test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()