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

SCRIPT_DIR = Path(__file__).parent.absolute()

class ChessDataset(Dataset):
    """Chess position dataset with normalization for wide evaluation range"""
    
    def __init__(self, json_file, max_eval=200.0):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.positions = []
        self.evaluations = []
        self.max_eval = max_eval
        
        # Track unique positions to avoid data leakage between train/val
        seen_keys = set()
        duplicate_count = 0
        clipped_count = 0
        total_count = len(data["positions"])
        
        for pos in data["positions"]:
            # Deduplication key: prefer FEN if available, otherwise hash the tensor
            if "fen" in pos["board"]:
                # Use EPD (first 4 fields of FEN) to ignore irrelevant move clocks
                key = " ".join(pos["board"]["fen"].split()[:4])
            else:
                # Fallback: create a tuple hash of the tensor
                key = tuple(pos["board"]["tensor"])
            
            if key in seen_keys:
                duplicate_count += 1
                continue
            seen_keys.add(key)
            
            # Explicitly use float32 for tensor data
            self.positions.append(torch.tensor(pos["board"]["tensor"], dtype=torch.float32).reshape(14, 8, 8))
            
            # Clip extreme evaluations to a reasonable range
            raw_eval = float(pos["evaluation"])
            clipped_eval = max(min(raw_eval, max_eval), -max_eval)
            
            if abs(raw_eval) > max_eval:
                clipped_count += 1
                
            # Normalize to [-1, 1] range and store as float32
            normalized_eval = torch.tensor(clipped_eval / max_eval, dtype=torch.float32)
            self.evaluations.append(normalized_eval)
        
        print(f"Loaded {len(self.positions)} unique positions ({duplicate_count} duplicates removed).")
        if clipped_count > 0:
            print(f"Notice: {clipped_count}/{total_count} positions ({100*clipped_count/total_count:.1f}%) had evaluations clipped to ±{max_eval}")
            
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
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out


class ChessNet(nn.Module):
    """Neural network for chess position evaluation with fixes for vanishing gradients"""
    
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Input: 14 channels (6 piece types * 2 colors + side to move + en passant)
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Three residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(3)
        ])
        
        # Value head with smaller layers to prevent overfitting
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Important: Use LeakyReLU to prevent dead neurons
        self.relu = nn.LeakyReLU(0.1)
        
        # Initialize with careful scaling to prevent vanishing gradients
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # More reasonable initialization for the final layer
                if m.out_features == 1:  # Output layer
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Value head
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 8 * 8)
        value = self.relu(self.value_fc1(value))
        
        # Apply tanh to squash the output to [-1, 1] range
        value = torch.tanh(self.value_fc2(value))
        
        return value


def train_model(dataset_path, output_model, epochs=500, batch_size=64, learning_rate=0.001, val_split=0.1, max_eval=2000.0):
    """Train the neural network with better handling of wide evaluation range"""
    
    # Load dataset with normalization
    full_dataset = ChessDataset(dataset_path, max_eval=max_eval)
    
    # Print dataset statistics
    evals = [y for _, y in full_dataset]
    print(f"Dataset stats: min={min(evals):.4f}, max={max(evals):.4f}, mean={sum(evals)/len(evals):.4f}")
    
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
    
    # Use MSE loss but apply tanh to model output to match expected range
    criterion = nn.MSELoss()
    
    # Use Adam with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Simpler learning rate scheduler 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5 
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
    
    # Verify model gets gradients in first batch
    model.train()
    first_batch = next(iter(train_loader))
    inputs, targets = first_batch
    inputs = inputs.to(device).float()
    targets = targets.to(device).float().view(-1, 1)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)  # No tanh needed here now
    loss.backward()
    
    # Check if gradients are flowing
    has_grad = False
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            max_grad = max(max_grad, grad_norm)
            if grad_norm > 0:
                has_grad = True
                print(f"Gradient detected in {name}: {grad_norm:.6f}")
    
    if not has_grad:
        print("WARNING: No gradients detected in first batch! Check model architecture.")
    else:
        print(f"Maximum gradient norm: {max_grad:.6f}")
    
    # Main training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            # Explicitly convert to float32 before moving to device
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Track gradient norms
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            epoch_grad_norm += grad_norm ** 0.5
            num_batches += 1
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_dataset)
        train_losses.append(train_loss)
        
        avg_grad_norm = epoch_grad_norm / num_batches if num_batches > 0 else 0
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Explicitly convert to float32 before moving to device
                inputs = inputs.to(torch.float32).to(device)
                targets = targets.to(torch.float32).to(device).view(-1, 1)
                
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item() * inputs.size(0)
        
        val_loss = running_val_loss / len(val_dataset)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print progress including gradient norm
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Grad Norm: {avg_grad_norm:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save state dict for Python
            model_info = {
                "state_dict": model.state_dict(),
                "max_eval": max_eval
            }
            torch.save(model_info, best_model_path)

            # Also save TorchScript version of the best model for C++ engine
            try:
                model.eval()
                # Move to CPU for tracing to ensure portability
                cpu_model = model.cpu()
                example_input = torch.randn(1, 14, 8, 8)
                traced_best = torch.jit.trace(cpu_model, example_input)
                torch.jit.save(traced_best, str(best_model_path.with_suffix('.jit.pt')))
                # Move back to original device
                model.to(device)
                print(f"New best model saved with validation loss {val_loss:.6f} (JIT version: {best_model_path.with_suffix('.jit.pt')})")
            except Exception as e:
                print(f"Warning: Could not save JIT version of best model: {e}")
                model.to(device)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model for export
    model.load_state_dict(torch.load(best_model_path)["state_dict"])
    model.eval()
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(Path(output_model).with_suffix('.png'))
    
    # Save model in TorchScript format for C++ inference
    # Explicitly move to CPU for tracing to ensure portability across different hardware setups
    model.cpu()
    model.eval()
    example_input = torch.randn(1, 14, 8, 8)
    traced_script_module = torch.jit.trace(model, example_input)
    torch.jit.save(traced_script_module, output_model)
    
    # Move model back to device if needed (though function is ending)
    model.to(device)
    
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
    default_dataset = SCRIPT_DIR.parent / 'model' / 'initial_dataset.json'
    default_output = SCRIPT_DIR.parent / 'model' / 'chess_model_iter_0.pt'
    parser.add_argument('--dataset', type=str, default=str(default_dataset), help='Path to dataset JSON file')
    parser.add_argument('--output', type=str, default=str(default_output), help='Output PyTorch model path')
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