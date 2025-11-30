"""
Stateful Generation Script for HOPE Model
Demonstrates true O(1) per-token generation with correct positional encoding.
"""
import torch
import tiktoken
import time
from config import HOPEConfig
from model import HOPE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print(f"Using device: {device}")
    
    # Load Model
    config = HOPEConfig()
    model = HOPE(config)
    model.to(device)
    model.eval()
    
    # Try to load weights if available
    try:
        checkpoint = torch.load("out/hope_best.pt", map_location=device)
        model.load_state_dict(checkpoint['model'])
        print("✓ Loaded trained weights.")
    except FileNotFoundError:
        print("⚠ No trained weights found. Using random initialization for demo.")

    tokenizer = tiktoken.get_encoding("gpt2")
    
    # -------------------------------------------------------------------------
    # Stateful Generation Demo
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # Explanation
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("Why This Works (Technical Details)")
    print("="*70)
    print("1. Stateful Memory: The Titans memory state persists across tokens")
    print("2. Correct Positions: Each token gets the RIGHT positional embedding")
    print("   - Token at position 100 gets pos_emb[100], not pos_emb[0]")
    print("3. O(1) Complexity: Each new token takes constant time")
    print("4. Infinite Context: Memory is constant-size regardless of history")
    print("\nThis is fundamentally different from standard Transformers:")
    print("- Transformer: O(T²) time, O(T) memory (KV cache)")
    print("- HOPE: O(1) time, O(1) memory (Titans state)")
    print("="*70)

if __name__ == "__main__":
    main()
