import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os
from collections import defaultdict

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out.reshape(-1, self.hidden_size))
        return out, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

def load_and_preprocess_data(base_folder, fine_tune_folder=None, language_filter=None):
    code_data = defaultdict(str)
    chars = set()
    
    def process_folder(folder):
        for root, _, files in os.walk(folder):
            for file in files:
                ext = file.lower()
                if ext.endswith(('.c', '.cpp', '.java', '.py')):
                    if ext.endswith(('.c', '.cpp')):
                        lang = 'c_family'
                    elif ext.endswith('.java'):
                        lang = 'java'
                    else:
                        lang = 'python'
                    
                    if language_filter and lang != language_filter:
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                            code_data[lang] += code
                            chars.update(code)
                    except UnicodeDecodeError:
                        continue
    
    process_folder(base_folder)
    if fine_tune_folder:
        process_folder(fine_tune_folder)
    
    if not code_data:
        return None, None, None  # Return None if no data is loaded
    
    chars = sorted(list(chars))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}

    encoded_data = {
        lang: [char_to_idx[c] for c in code if c in char_to_idx]
        for lang, code in code_data.items()
    }

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

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, 
                device, num_epochs, sequence_length, vocab_size, save_path,
                fine_tuning=False):
    best_test_accuracy = 0
    patience = 5  # Early stopping patience
    no_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_predictions = 0
        total_loss = 0
        
        for i, batch in enumerate(train_dataloader):
            batch_size = batch.size(0)
            hidden = model.init_hidden(batch_size, device)
            
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:].reshape(-1)
            
            hidden = tuple([h.detach() for h in hidden])
            outputs, hidden = model(inputs, hidden)
            outputs = outputs.view(batch_size * (sequence_length - 1), vocab_size)
            
            loss = criterion(outputs, targets)
            
            # Add L2 regularization during fine-tuning
            if fine_tuning:
                l2_lambda = 0.01
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
                
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == targets).sum().item()
            total_predictions += targets.size(0)
            total_loss += loss.item()
            
            if i % 100 == 0:
                avg_loss = total_loss / (i + 1)
                train_accuracy = total_correct / total_predictions * 100
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], "
                      f"Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, 
                                                  device, sequence_length)
        print(f"Epoch [{epoch+1}/{num_epochs}] Test Loss: {test_loss:.4f}, "
              f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Save best model and implement early stopping
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            no_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_accuracy,
            }, save_path)
            print(f"Saved new best model with test accuracy: {test_accuracy:.2f}%")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping triggered")
                break

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
    hidden_size = 512
    num_layers = 2
    dropout = 0.2
    learning_rate = 0.001
    sequence_length = 100
    batch_size = 64
    test_size = 0.2
    
    # Base training parameters
    base_epochs = 5
    
    # Fine-tuning parameters
    fine_tune_epochs = 3
    fine_tune_lr = 0.0001

    # Paths
    base_folder = "C:\\Projects\\Major_ML\\New_Db"
    fine_tune_folder = "C:\\Projects\\Major_ML\\C_Programs"
    base_model_path = "C:\\Projects\\Major_ML\\Saved_Path\\base_model.pth"
    fine_tuned_model_path = "C:\\Projects\\Major_ML\\Saved_Path\\fine_tuned_model.pth"

    # Load and prepare base training data
    encoded_data, char_to_idx, idx_to_char = load_and_preprocess_data(base_folder)
    vocab_size = len(char_to_idx)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = CharRNN(vocab_size, hidden_size, vocab_size, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Base training
    print("Starting base training...")
    all_sequences = []
    for lang_data in encoded_data.values():
        all_sequences.extend(create_sequences(lang_data, sequence_length))
    
    train_sequences, test_sequences = train_test_split(all_sequences, 
                                                     test_size=test_size, 
                                                     random_state=42)
    
    train_dataloader = create_dataloader(train_sequences, batch_size)
    test_dataloader = create_dataloader(test_sequences, batch_size, shuffle=False)
    
    train_model(model, train_dataloader, test_dataloader, criterion, optimizer, 
                device, base_epochs, sequence_length, vocab_size, base_model_path)

    # Fine-tuning
    if fine_tune_folder:
        print("Starting fine-tuning...")
        
        model.load_state_dict(torch.load(base_model_path, weights_only=True)['model_state_dict'])

        optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
        
        fine_tune_data, _, _ = load_and_preprocess_data(fine_tune_folder)
        
        if fine_tune_data is None:
            print("No data found in fine-tune folder. Skipping fine-tuning.")
            return
        
        fine_tune_sequences = []
        for lang_data in fine_tune_data.values():
            fine_tune_sequences.extend(create_sequences(lang_data, sequence_length))
        
        fine_tune_train_sequences, fine_tune_test_sequences = train_test_split(
            fine_tune_sequences, test_size=test_size, random_state=42
        )
        
        fine_tune_train_dataloader = create_dataloader(fine_tune_train_sequences, batch_size)
        fine_tune_test_dataloader = create_dataloader(fine_tune_test_sequences, batch_size, shuffle=False)
        
        train_model(model, fine_tune_train_dataloader, fine_tune_test_dataloader, criterion, optimizer,
                    device, fine_tune_epochs, sequence_length, vocab_size, fine_tuned_model_path,
                    fine_tuning=True)

if __name__ == "__main__":
    main()
