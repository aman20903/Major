import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import os

class ImprovedCharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # LayerNorm for hidden size, adjusted to the correct dimension
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if self.bidirectional else 1))
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if self.bidirectional else 1), output_size)

        # Add L2 regularization (weight decay) to the fully connected layer
        self.fc.weight.data.normal_(0.0, 1.0 / np.sqrt(self.fc.in_features))
        self.fc.bias.data.zero_()
        self.weight_decay = weight_decay

    def forward(self, x, hidden):
        batch_size = x.size(0)
        seq_length = x.size(1)

        x = self.embedding(x)
        
        # LSTM output: batch_size x seq_length x hidden_size * num_directions
        lstm_out, hidden = self.lstm(x, hidden)
        
        # If bidirectional, concatenate the forward and backward states
        if self.bidirectional:
            lstm_out = lstm_out.contiguous().view(batch_size, seq_length, 2, self.hidden_size)
            lstm_out = lstm_out.permute(0, 2, 1, 3).contiguous()
            lstm_out = lstm_out.view(batch_size * seq_length, self.hidden_size * 2)
        else:
            lstm_out = lstm_out.contiguous().view(batch_size * seq_length, self.hidden_size)
        
        # Apply LayerNorm after reshaping the LSTM output
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout_layer(lstm_out)
        
        output = self.fc(lstm_out)
        
        return output, hidden

    def init_hidden(self, batch_size, device):
        num_directions = 2 if self.bidirectional else 1
        return (torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(device))

def load_codeforces_data(folder_path):
    code_data = ""
    chars = set()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.java', '.py', '.cpp')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        code_data += code
                        chars.update(code)
                except UnicodeDecodeError:
                    continue

    chars = sorted(list(chars))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    encoded_data = [char_to_idx[c] for c in code_data if c in char_to_idx]

    return encoded_data, char_to_idx, idx_to_char

def create_sequences(encoded_data, sequence_length):
    sequences = []
    stride = sequence_length // 2
    for i in range(0, len(encoded_data) - sequence_length, stride):
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
            inputs = batch[:, :-1]
            targets = batch[:, 1:].reshape(-1)
            
            outputs, _ = model(inputs, hidden)
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
    hidden_size = 512  # Increased hidden size
    num_layers = 3  # Increased number of layers
    num_epochs = 10  # Increased number of epochs
    learning_rate = 0.0005  # Reduced learning rate for finer adjustments
    sequence_length = 100
    batch_size = 32
    test_size = 0.2
    dropout = 0.4  # Increased dropout to prevent overfitting
    weight_decay = 1e-5  # Reduced weight decay

    # Load and prepare data
    encoded_data, char_to_idx, idx_to_char = load_codeforces_data("C:\\Projects\\Major_ML\\New_Db")
    vocab_size = len(char_to_idx)
    
    sequences = create_sequences(encoded_data, sequence_length)
    train_sequences, val_sequences, test_sequences, _ = train_test_split(sequences, sequences, test_size=test_size, random_state=42)
    
    train_dataloader = create_dataloader(train_sequences, batch_size)
    val_dataloader = create_dataloader(val_sequences, batch_size, shuffle=False)
    test_dataloader = create_dataloader(test_sequences, batch_size, shuffle=False)
    
    # Initialize model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ImprovedCharRNN(vocab_size, hidden_size, vocab_size, num_layers, dropout=dropout, weight_decay=weight_decay).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training loop
    best_val_accuracy = 0
    patience = 5
    patience_counter = 0
    max_grad_norm = 1.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_predictions = 0
        
        for i, batch in enumerate(train_dataloader):
            batch_size = batch.size(0)
            hidden = model.init_hidden(batch_size, device)
            
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:].reshape(-1)
            
            # Forward pass
            hidden = tuple([h.detach() for h in hidden])
            outputs, hidden = model(inputs, hidden)
            
            # Calculate loss and accuracy
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == targets).sum().item()
            total_predictions += targets.size(0)
            
            if i % 100 == 0:
                avg_loss = total_loss / (i + 1)
                accuracy = (total_correct / total_predictions) * 100
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], "
                      f"Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
        
        # Evaluate on validation set
        val_loss, val_accuracy = evaluate_model(model, val_dataloader, criterion, device, sequence_length)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Save best model and check early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char
            }, "C:\\Projects\\Major_ML\\best_model.pth")
            print("Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break
    
    # Evaluate on test set
    test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device, sequence_length)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
