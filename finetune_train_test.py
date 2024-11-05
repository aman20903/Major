import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch.nn.utils import clip_grad_norm_
from typing import Tuple, Dict, List
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self):
        self.input_size = 128
        self.hidden_size = 512
        self.num_layers = 2
        self.num_epochs = 5
        self.learning_rate = 0.001
        self.sequence_length = 100
        self.batch_size = 64
        self.clip_grad = 5.0
        self.dropout = 0.5
        self.save_path = "char_rnn_model.pth"

class DataProcessor:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.code_data = ""
        self.chars = set()
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        
    def load_and_process_data(self) -> Tuple[List[int], Dict[str, int], Dict[int, str]]:
        """Load and process code files from the specified folder."""
        file_count = 0
        total_chars = 0
        
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(('.java', '.py', '.cpp')):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                            if code:
                                file_count += 1
                                total_chars += len(code)
                                self.code_data += code
                                self.chars.update(code)
                                if file_count % 100 == 0:
                                    logging.info(f"Processed {file_count} files, total chars: {total_chars}")
                    except Exception as e:
                        logging.error(f"Error processing file {file_path}: {str(e)}")

        if not self.code_data:
            raise ValueError("No valid code files found in the specified directory")

        self.chars = sorted(list(self.chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        encoded_data = [self.char_to_idx[c] for c in self.code_data if c in self.char_to_idx]
        return encoded_data, self.char_to_idx, self.idx_to_char

class CharRNN(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super(CharRNN, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        
        self.embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out.reshape(-1, self.hidden_size))
        return out, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

class ModelTrainer:
    def __init__(self, config: Config, model: CharRNN, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def create_dataloader(self, data: List[int]) -> torch.utils.data.DataLoader:
        sequences = []
        for i in range(0, len(data) - self.config.sequence_length, self.config.sequence_length):
            seq = torch.tensor(data[i:i + self.config.sequence_length], dtype=torch.long)
            sequences.append(seq)
        return torch.utils.data.DataLoader(sequences, batch_size=self.config.batch_size, shuffle=True)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader, phase: str) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_predictions = 0
        start_time = time.time()

        for i, batch in enumerate(dataloader):
            batch_size = batch.size(0)
            hidden = self.model.init_hidden(batch_size, self.device)
            batch = batch.to(self.device)
            
            inputs, targets = batch[:, :-1], batch[:, 1:].reshape(-1)
            hidden = tuple([h.detach() for h in hidden])
            
            self.optimizer.zero_grad()
            outputs, hidden = self.model(inputs, hidden)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == targets).sum().item()
            total_predictions += targets.size(0)
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                logging.info(f"{phase} Batch [{i+1}/{len(dataloader)}], "
                           f"Loss: {loss.item():.4f}, "
                           f"Time: {elapsed:.2f}s")
                
        avg_loss = total_loss / len(dataloader)
        accuracy = (total_correct / total_predictions) * 100
        return avg_loss, accuracy

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initial training
    try:
        data_processor = DataProcessor("C:\\Users\\kulam\\OneDrive\\Documents\\Codeforces_Dataset\\New_Db")
        encoded_data, char_to_idx, idx_to_char = data_processor.load_and_process_data()
        vocab_size = len(char_to_idx)
        
        model = CharRNN(config, vocab_size).to(device)
        trainer = ModelTrainer(config, model, device)
        
        train_data, _ = train_test_split(encoded_data, test_size=0.2, random_state=42)
        train_loader = trainer.create_dataloader(train_data)
        
        for epoch in range(config.num_epochs):
            avg_loss, accuracy = trainer.train_epoch(train_loader, "Training")
            logging.info(f"Epoch [{epoch+1}/{config.num_epochs}], "
                        f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'vocab': char_to_idx,
                'config': config.__dict__
            }
            torch.save(checkpoint, config.save_path)
            
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()