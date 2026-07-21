"""
Training Script for HOPE Model (Colab Free-Tier Optimized)

Features:
- Automatic Google Drive mounting and checkpoint auto-resume
- Stateful memory persistence across training batches
- Mixed precision (fp16 autocast + GradScaler)
- Continuum Memory System (CMS) multi-rate gradient updates
- Sample generation logging at every evaluation step
- Scaled for ~300M - 1B tokens total across multiple resumed sessions
"""
import os
import time
import math
import torch
import tiktoken
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import autocast, GradScaler
from config import HOPEConfig
from model import HOPE

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
batch_size = 32
max_iters = 50000       # ~400M tokens total (32 batch * 256 block * 50k steps)
eval_interval = 250
save_interval = 250
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 1000
grad_clip = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
state_reset_interval = 5000  # Long reset interval to maintain continuous memory
out_dir = 'out'
# -----------------------------------------------------------------------------

def setup_checkpoint_dir(out_dir='out'):
    """Mount Google Drive if available and create checkpoint directories."""
    drive_dir = '/content/drive/MyDrive/hope_checkpoints'
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
        os.makedirs(drive_dir, exist_ok=True)
        print(f"✓ Google Drive checkpoint directory: {drive_dir}")
        return drive_dir
    except Exception:
        os.makedirs(out_dir, exist_ok=True)
        print(f"✓ Local checkpoint directory: {out_dir}")
        return out_dir

class StreamingTextDataset(IterableDataset):
    """Memory-efficient streaming dataset from TinyStories"""
    def __init__(self, split="train", block_size=256):
        from datasets import load_dataset
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

def generate_sample(model, tokenizer, prompt="Once upon a time,", max_new_tokens=50):
    """Generates a text sample for evaluation logging"""
    model.eval()
    start_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out_ids = model.generate(start_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    sample_text = tokenizer.decode(out_ids[0].tolist())
    model.train()
    return sample_text

def estimate_loss(model, val_loader, persistent_states):
    """Evaluate validation loss with state persistence"""
    model.eval()
    losses = torch.zeros(eval_iters)
    eval_states = persistent_states
    
    for k, (X, Y) in enumerate(val_loader):
        if k >= eval_iters:
            break
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            with autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16 if device == 'cuda' else torch.float32):
                logits, loss, eval_states = model(X, Y, states=eval_states)
            eval_states = [s.detach() if s is not None else None for s in eval_states]
        losses[k] = loss.item()
    
    model.train()
    return losses.mean().item()

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
    ckpt_dir = setup_checkpoint_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    config = HOPEConfig()
    model = HOPE(config).to(device)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scaler = GradScaler('cuda', enabled=(device == 'cuda'))
    
    # Auto-resume logic
    latest_ckpt_path = os.path.join(ckpt_dir, "hope_latest.pt")
    local_latest_ckpt = os.path.join(out_dir, "hope_latest.pt")
    
    ckpt_to_load = None
    if os.path.exists(latest_ckpt_path):
        ckpt_to_load = latest_ckpt_path
    elif os.path.exists(local_latest_ckpt):
        ckpt_to_load = local_latest_ckpt
        
    start_iter = 0
    best_val_loss = 1e9
    persistent_states = None
    
    if ckpt_to_load:
        print(f"Loading checkpoint from {ckpt_to_load}...")
        checkpoint = torch.load(ckpt_to_load, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint.get('best_val_loss', 1e9)
        persistent_states = checkpoint.get('persistent_states', None)
        print(f"✓ Resumed training from step {start_iter} (best val loss: {best_val_loss:.4f})")
        
    train_dataset = StreamingTextDataset(split="train", block_size=config.block_size)
    val_dataset = StreamingTextDataset(split="validation", block_size=config.block_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_iter = iter(train_loader)
    
    print("\n" + "="*60)
    print("HOPE MODEL TRAINING (STATEFUL MEMORY + CMS MULTI-RATE)")
    print(f"Target steps: {max_iters} | Batch size: {batch_size} | Block size: {config.block_size}")
    print("="*60 + "\n")
    
    t0 = time.time()
    iter_num = start_iter
    
    while iter_num < max_iters:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
            
        X, Y = X.to(device), Y.to(device)
        
        # Forward pass with mixed precision and state persistence
        with autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
            logits, loss, new_states = model(X, Y, states=persistent_states)
            
        # Detach states to bound backpropagation computation graph
        persistent_states = [s.detach() if s is not None else None for s in new_states]
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Enforce multi-rate updates for CMS blocks
        scaler.unscale_(optimizer)
        model.enforce_cms_update_periods(iter_num)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Periodic state reset to prevent drift
        if iter_num > 0 and iter_num % state_reset_interval == 0:
            print(f"[Step {iter_num}] Periodic state reset")
            persistent_states = None
            
        # Evaluation & Checkpoint saving
        if iter_num % eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, persistent_states)
            dt = time.time() - t0
            t0 = time.time()
            print(f"\n--- Step {iter_num}/{max_iters} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e} | Time: {dt:.2f}s ---")
            
            # Generate sample
            sample_text = generate_sample(model, tokenizer)
            print(f"Sample: \"{sample_text.strip()}\"\n")
            
            # Save latest checkpoint
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'persistent_states': persistent_states,
                'config': config,
            }
            
            torch.save(checkpoint, os.path.join(ckpt_dir, "hope_latest.pt"))
            torch.save(checkpoint, os.path.join(out_dir, "hope_latest.pt"))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint['best_val_loss'] = best_val_loss
                torch.save(checkpoint, os.path.join(ckpt_dir, "hope_best.pt"))
                torch.save(checkpoint, os.path.join(out_dir, "hope_best.pt"))
                print(f"✓ Saved new best model (val_loss: {best_val_loss:.4f})")
                
        iter_num += 1

    print(f"\nTraining complete! Final best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
