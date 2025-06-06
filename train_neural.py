#!/usr/bin/env python3
"""
Neural network training for TDChess
This script trains a neural network on chess positions and exports it to ONNX format
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ChessDataset(Dataset):
    """Chess position dataset"""
    
    def __init__(self, json_file):
        """
        Args:
            json_file (string): Path to the JSON file with positions
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.positions = []
        self.evaluations = []
        
        for pos in data['positions']:
            tensor = np.array(pos['board']['tensor'], dtype=np.float32)
            # Reshape from flat to 4D: [batch, channels, height, width]
            tensor = tensor.reshape(1, 14, 8, 8)
            self.positions.append(tensor)
            self.evaluations.append(pos['evaluation'])
        
        # Convert to numpy arrays
        self.positions = np.vstack(self.positions)
        self.evaluations = np.array(self.evaluations, dtype=np.float32)
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]


class ChessNet(nn.Module):
    """Neural network for chess position evaluation"""
    
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(64) for _ in range(5)
        ])
        
        # Policy head (not used for evaluation, but useful for training)
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1968)  # 1968 possible moves in chess
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x += residual
            x = nn.functional.relu(x)
        
        # Only return value for evaluation
        return self.value_head(x)


def train_model(dataset_path, output_model, epochs=100, batch_size=64, learning_rate=0.001):
    """Train the neural network and export to ONNX"""
    
    # Load dataset
    dataset = ChessDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = ChessNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for positions, evaluations in dataloader:
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
            
            running_loss += loss.item()
        
        # Print statistics
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.6f}")
    
    # Export to ONNX
    model.eval()
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
    return model


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train a neural network for chess evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--output', type=str, default='chess_model.onnx', help='Output ONNX model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
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