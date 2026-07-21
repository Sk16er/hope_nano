"""
HOPE Model Stateful Inference Script

Demonstrates O(1) per-token generation by passing a persistent, constant-size
memory state instead of a growing KV cache.
"""
import os
import time
import torch
import tiktoken
from config import HOPEConfig
from model import HOPE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_explanation():
    separator = f"{'='*15} TECHNICAL EXPLANATION {'='*15}"
    print("\n" + separator)
    print("\n[HOPE ARCHITECTURE ADVANTAGES]")
    print(f"{' '*4}• Stateful Memory: Titans memory matrix persists across chunks & tokens.")
    print(f"{' '*4}• Data-Dependent Gating: Alpha(x) & Beta(x) regulate forget/write rates per token.")
    print(f"{' '*4}• CMS Hierarchy: Multi-rate MLP blocks update at different timescales.")
    print(f"{' '*4}• O(1) Time & Memory Decoding: Each new token generated in constant time.")
    print("\n[CONTRAST WITH STANDARD TRANSFORMERS]")
    print(f"{' '*4}• Transformer: O(T²) time, O(T) memory (growing KV cache)")
    print(f"{' '*4}• HOPE: O(1) time, O(1) memory (fixed Titans state)")
    print(f"{'='*len(separator)}\n")

def main():
    print(f"Using device: {device}")
    
    config = HOPEConfig()
    model = HOPE(config).to(device).eval()
    
    ckpt_path = "out/hope_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "out/hope_latest.pt"
        
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"✓ Loaded trained weights from {ckpt_path}.")
    except FileNotFoundError:
        print("⚠ No trained weights found. Running with random initialization.")

    tokenizer = tiktoken.get_encoding("gpt2")
    
    prompts = [
        "Once upon a time, in a land far away,",
        "The secret to artificial intelligence is",
        "In the year 2050, humanity discovered",
    ]
    
    for prompt in prompts:
        print("\n" + "="*70)
        print(f"Prompt: {prompt}")
        print("="*70)
        
        start_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        t0 = time.time()
        output_ids = model.generate(start_ids, max_new_tokens=150, temperature=0.8, top_k=40)
        t1 = time.time()
        
        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(generated_text)
        print(f"\n⏱ Time: {t1 - t0:.2f}s | Speed: {150 / (t1 - t0):.1f} tokens/sec")
    
    print_explanation()

if __name__ == "__main__":
    main()
