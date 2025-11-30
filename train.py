"""
Training Script for HOPE Model (Production-Grade)
CRITICAL: This script maintains memory states across batches to enable true continual learning.
"""
import os
import time
import math
import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
from config import HOPEConfig
from model import HOPE

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
batch_size = 8  # Reduced for stability with state persistence
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 200
grad_clip = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
out_dir = 'out'
state_reset_interval = 500  # Reset states periodically to prevent drift
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)

class StreamingTextDataset(IterableDataset):
    """Memory-efficient streaming dataset"""
    def __init__(self, split="train", block_size=256):
        self.dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
    
    def __iter__(self):
        buffer = []
        for item in self.dataset:
            tokens = self.tokenizer.encode(item['text'])
            buffer.extend(tokens)
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

def estimate_loss(model, val_loader, persistent_states):
    """Evaluate with state persistence"""
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    eval_states = persistent_states  # Start from current training state
    
    for k, (X, Y) in enumerate(val_loader):
        if k >= eval_iters:
            break
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            logits, loss, eval_states = model(X, Y, states=eval_states)
            # Detach states
            eval_states = [s.detach() if s is not None else None for s in eval_states]
        losses[k] = loss.item()
    
    out['val'] = losses.mean()
    model.train()
    return out

def get_lr(it):
    """Cosine learning rate schedule with warmup"""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def main():
    print(f"Using device: {device}")
    
    config = HOPEConfig()
    config.block_size = 256
    
    model = HOPE(config)
    model.to(device)
    
    # CRITICAL FIX: Compile the model
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("✓ Compilation enabled (6-10x speedup)")
    else:
        print("⚠ torch.compile not available (update to PyTorch 2.0+)")
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scaler = GradScaler()  # Mixed precision
    
    train_dataset = StreamingTextDataset(split="train", block_size=config.block_size)
    val_dataset = StreamingTextDataset(split="validation", block_size=config.block_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    iter_num = 0
    best_val_loss = 1e9
    t0 = time.time()
    
    # CRITICAL FIX: Persistent states across batches
    persistent_states = None
    
    train_iter = iter(train_loader)
    
    print("\n" + "="*60)
    print("TRAINING WITH STATEFUL MEMORY")
    print("="*60)
    print("Memory states will persist across batches.")
    print(f"State reset interval: {state_reset_interval} steps")
    print("="*60 + "\n")
    
    while iter_num < max_iters:
        # Learning rate schedule
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
            
        X, Y = X.to(device), Y.to(device)
        
        # CRITICAL FIX: Forward pass with state persistence
        with autocast():
            logits, loss, new_states = model(X, Y, states=persistent_states)
        
        # CRITICAL FIX: Detach states to prevent backprop through entire history
        persistent_states = [s.detach() if s is not None else None for s in new_states]
        
        # Backward pass with gradient scaling
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Periodic state reset to prevent drift
        if iter_num % state_reset_interval == 0 and iter_num > 0:
            print(f"[Step {iter_num}] Resetting states to prevent drift")
            persistent_states = None
        
        # Evaluation
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model, val_loader, persistent_states)
            print(f"step {iter_num}: train loss {loss.item():.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                # Save model checkpoint
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(out_dir, "hope_best.pt"))
                print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
            
        iter_num += 1
        
    print(f"\n{'='*60}")
    print(f"Training finished in {time.time() - t0:.2f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
