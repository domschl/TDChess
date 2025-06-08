#!/usr/bin/env python3
"""
Neural network training for TDChess using PyTorch
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

class ChessDataset(Dataset):
    """Chess position dataset"""
    
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.positions = []
        self.evaluations = []
        
        for pos in data["positions"]:
            self.positions.append(torch.tensor(pos["board"]["tensor"], dtype=torch.float32).reshape(14, 8, 8))
            # Explicitly convert evaluation to float32
            self.evaluations.append(torch.tensor(float(pos["evaluation"]), dtype=torch.float32))
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out


class ChessNet(nn.Module):
    """Improved neural network for chess position evaluation"""
    
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Input: 14 channels (6 piece types * 2 colors + side to move + en passant)
        # First block
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(5)  # Add 5 residual blocks
        ])
        
        # Value head
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Proper weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.bn1(self.relu(self.conv1(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Value head
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 8 * 8)
        value = self.relu(self.value_fc1(value))
        value = self.tanh(self.value_fc2(value))
        
        return value


def train_model(dataset_path, output_model, epochs=500, batch_size=64, learning_rate=0.001, val_split=0.1):
    """Train the neural network with improved convergence"""
    
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
    
    # Higher initial learning rate
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Better learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate*10,  # Peak at 10x the base learning rate
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warm up for 10% of training
        div_factor=10.0,  # Start with lr/10
        final_div_factor=100.0  # End with lr/1000
    )
    
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
    
    # Track gradients and weights to detect problems
    grad_norms = []
    weight_norms = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_weight_norm = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            # Convert to float32 before moving to device
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device).view(-1, 1)
            
            # Scale targets - this can help convergence if evaluations are in centipawns
            targets = targets / 100.0  # Scale down if evaluations are in centipawns
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Track gradients and weights
            total_grad_norm = 0.0
            total_weight_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_weight_norm += p.data.norm(2).item() ** 2
            
            epoch_grad_norm += total_grad_norm ** 0.5
            epoch_weight_norm += total_weight_norm ** 0.5
            num_batches += 1
            
            optimizer.step()
            scheduler.step()  # Step per batch with OneCycleLR
            
            running_loss += loss.item() * inputs.size(0)
        
        # Average norms for the epoch
        grad_norms.append(epoch_grad_norm / num_batches)
        weight_norms.append(epoch_weight_norm / num_batches)
        
        train_loss = running_loss / len(train_dataset)
        train_losses.append(train_loss)
        
        # Print debug info about gradients and weights
        if epoch % 5 == 0:
            print(f"Gradient norm: {grad_norms[-1]:.6f}, Weight norm: {weight_norms[-1]:.6f}")
        
        # Validation phase with the same target scaling
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(torch.float32).to(device)
                targets = targets.to(torch.float32).to(device).view(-1, 1)
                targets = targets / 100.0  # Apply the same scaling
                
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item() * inputs.size(0)
        
        val_loss = running_val_loss / len(val_dataset)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"New best model saved with validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
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
    
    # Save model in TorchScript format for C++ inference
    dummy_input = torch.randn(1, 14, 8, 8, device=device)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(output_model)
    
    print(f"Model exported to {output_model}")
    
    # Verify model with a sample evaluation
    if len(full_dataset) > 0:
        sample_input, sample_target = full_dataset[0]
        sample_input = sample_input.unsqueeze(0).to(device)
        with torch.no_grad():
            sample_output = model(sample_input)
        print(f"Sample evaluation - Target: {sample_target:.6f}, Model output: {sample_output.item():.6f}")
    
    return model


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train a neural network for chess evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--output', type=str, default='chess_model.pt', help='Output PyTorch model path')
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