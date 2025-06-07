#!/usr/bin/env python3
# filepath: /Users/dsc/Codeberg/TDChess/train_neural.py
"""
Improved neural network training for TDChess
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path
import time

class ChessDataset(Dataset):
    """Chess position dataset"""
    
    def __init__(self, json_file):
        """
        Args:
            json_file (string): Path to the JSON file with positions
        """
        print(f"Loading dataset from {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.positions = []
        self.evaluations = []
        self.fens = []
        
        for pos in data['positions']:
            tensor = np.array(pos['board']['tensor'], dtype=np.float32)
            # Reshape from flat to 4D: [channels, height, width]
            tensor = tensor.reshape(14, 8, 8)
            self.positions.append(tensor)
            self.evaluations.append(pos['evaluation'])
            self.fens.append(pos['board']['fen'])
        
        # Convert to numpy arrays
        self.positions = np.array(self.positions, dtype=np.float32)
        self.evaluations = np.array(self.evaluations, dtype=np.float32)
        
        # Normalize evaluations to a reasonable range for tanh activation
        # Typical chess engines use centipawns, so divide by 100 to get pawn units
        self.evaluations_raw = self.evaluations.copy()
        self.evaluations = np.clip(self.evaluations / 100.0, -1.0, 1.0)
        
        print(f"Loaded {len(self.positions)} positions")
        print(f"Evaluation stats - Min: {self.evaluations.min():.2f}, Max: {self.evaluations.max():.2f}, Mean: {self.evaluations.mean():.2f}")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]


class ChessNet(nn.Module):
    """Improved neural network for chess position evaluation"""
    
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # First convolutional block with batch norm
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Second convolutional block
        self.conv_block21 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_block22 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_block23 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_block24 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        
        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Add dropout to prevent overfitting
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block21(x)
        x = self.conv_block22(x)
        x = self.conv_block23(x)
        x = self.conv_block24(x)
        x = self.conv_block3(x)
        return self.value_head(x)


def train_model(dataset_path, output_model, epochs=500, batch_size=64, learning_rate=0.001, val_split=0.1):
    """Train the neural network and export to ONNX"""
    
    # Load dataset
    full_dataset = ChessDataset(dataset_path)
    
    # Split into training and validation sets
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Training set: {len(train_dataset)} positions")
    print(f"Validation set: {len(val_dataset)} positions")
    
    # Create model
    model = ChessNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    print(f"Training on {device} for {epochs} epochs...")
    
    # For tracking progress
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_path = Path(output_model).with_suffix('.best.pt')
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for positions, evaluations in train_loader:
            positions = positions.to(device)
            evaluations = evaluations.to(device).unsqueeze(1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(positions)
            loss = criterion(outputs, evaluations)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * positions.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for positions, evaluations in val_loader:
                positions = positions.to(device)
                evaluations = evaluations.to(device).unsqueeze(1)
                outputs = model(positions)
                loss = criterion(outputs, evaluations)
                val_loss += loss.item() * positions.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Print statistics
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | {epoch_time:.1f}s | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model for export
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('log(Loss)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.yscale('log')
    plt.savefig(Path(output_model).with_suffix('.png'))
    
    # Export to ONNX
    dummy_input = torch.randn(1, 14, 8, 8, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_model,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported to {output_model}")
    
    # Verify model with a sample evaluation
    if len(full_dataset) > 0:
        sample_idx = np.random.randint(0, len(full_dataset))
        sample_pos = torch.tensor(full_dataset.positions[sample_idx:sample_idx+1]).to(device)
        sample_eval = full_dataset.evaluations_raw[sample_idx]
        
        with torch.no_grad():
            model_output = model(sample_pos).item() * 100  # Scale back to centipawns
        
        print("\nSample evaluation:")
        print(f"FEN: {full_dataset.fens[sample_idx]}")
        print(f"Ground truth: {sample_eval:.2f}")
        print(f"Model prediction: {model_output:.2f}")
    
    return model


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train a neural network for chess evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--output', type=str, default='chess_model.onnx', help='Output ONNX model path')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Initial learning rate')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == '__main__':
    main()