"""
HOPE Model Stateful Inference Script.

Demonstrates O(1) per-token generation by passing a persistent, constant-size
memory state instead of a growing KV cache.
"""
import torch
import tiktoken
import time
from config import HOPEConfig
from model import HOPE # Assumes HOPE and HOPEConfig are available

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_explanation():
    # Prints the technical explanation of HOPE's efficiency.
    separator = f"{'='*15} TECHNICAL EXPLANATION {'='*15}"
    print("\n" + separator)
    
    print("\n[HOPE ARCHITECTURE ADVANTAGES]")
    
    print(f"\n{' '*4}• Stateful Memory: Titans memory persists across tokens.")
    print(f"{' '*4}• O(1) Complexity: Each new token takes constant time.")
    print(f"{' '*4}• Infinite Context: Memory size is constant.")
    
    print("\n[CONTRAST WITH TRANSFORMERS]")
    print(f"\n{' '*4}• Transformer: O(T²) time, O(T) memory (KV cache)")
    print(f"{' '*4}• HOPE: O(1) time, O(1) memory (Titans state)")
    
    print(f"{'='*len(separator)}\n")


def main():
    print(f"Using device: {device}")
    
    config = HOPEConfig()
    model = HOPE(config).to(device).eval()
    
    # Load trained model weights from checkpoint
    try:
        checkpoint = torch.load("out/hope_best.pt", map_location=device)
        model.load_state_dict(checkpoint['model'])
        print("✓ Loaded trained weights.")
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
        output_ids = model.generate(start_ids, max_new_tokens=200, temperature=0.8, top_k=50)
        t1 = time.time()
        
        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(generated_text)
        print(f"\n⏱ Time: {t1 - t0:.2f}s | Tokens/sec: {200 / (t1 - t0):.1f}")
    
    print_explanation()

if __name__ == "__main__":
    main()
