"""
Unit test for TitansL2 module in HOPE architecture.
Verifies that token-by-token forward_inference matches forward_train_chunkwise
output and final state within floating point tolerance.
"""
import torch
import torch.nn.functional as F
from config import HOPEConfig
from model import TitansL2, HOPE

def test_titans_equivalence():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = HOPEConfig(n_embd=256, n_head=4, block_size=256)
    titans = TitansL2(config).to(device).eval()
    
    B = 2
    T = 64
    x = torch.randn(B, T, config.n_embd, device=device)
    head_dim = config.n_embd // config.n_head
    
    # Test 1: Equivalence with state=None (zero initial state)
    print("Testing TitansL2 equivalence with state=None...")
    with torch.no_grad():
        y_chunkwise, state_chunkwise = titans(x, state=None)
        
        # Token-by-token forward_inference manual scan
        q = titans.c_q(x).view(B, T, titans.n_head, head_dim).transpose(1, 2)
        k = titans.c_k(x).view(B, T, titans.n_head, head_dim).transpose(1, 2)
        v = titans.c_v(x).view(B, T, titans.n_head, head_dim).transpose(1, 2)
        k = F.normalize(k, dim=-1)
        
        alpha = torch.sigmoid(titans.c_alpha(x)) * 0.5
        beta = torch.sigmoid(titans.c_beta(x)) * 0.5
        
        curr_state = torch.zeros(B, titans.n_head, head_dim, head_dim, device=device)
        ys = []
        
        for t in range(T):
            qt = q[:, :, t:t+1, :]
            kt = k[:, :, t:t+1, :]
            vt = v[:, :, t:t+1, :]
            at = alpha[:, t:t+1, :]
            bt = beta[:, t:t+1, :]
            yt, curr_state = titans.forward_inference(qt, kt, vt, at, bt, curr_state)
            ys.append(yt)
            
        y_stepwise = torch.cat(ys, dim=1)
        
    diff_y = (y_chunkwise - y_stepwise).abs().max().item()
    diff_state = (state_chunkwise - curr_state).abs().max().item()
    print(f"  - Output Max Diff: {diff_y:.6e}")
    print(f"  - State  Max Diff: {diff_state:.6e}")
    
    assert torch.allclose(y_chunkwise, y_stepwise, atol=1e-4, rtol=1e-4), f"Outputs mismatch: {diff_y}"
    assert torch.allclose(state_chunkwise, curr_state, atol=1e-4, rtol=1e-4), f"States mismatch: {diff_state}"
    print("✓ Test 1 Passed (state=None)!")
    
    # Test 2: Equivalence with non-zero initial state
    print("\nTesting TitansL2 equivalence with non-zero initial state...")
    initial_state = torch.randn(B, titans.n_head, head_dim, head_dim, device=device)
    
    with torch.no_grad():
        y_chunkwise2, state_chunkwise2 = titans(x, state=initial_state)
        
        curr_state2 = initial_state.clone()
        ys2 = []
        
        for t in range(T):
            qt = q[:, :, t:t+1, :]
            kt = k[:, :, t:t+1, :]
            vt = v[:, :, t:t+1, :]
            at = alpha[:, t:t+1, :]
            bt = beta[:, t:t+1, :]
            yt, curr_state2 = titans.forward_inference(qt, kt, vt, at, bt, curr_state2)
            ys2.append(yt)
            
        y_stepwise2 = torch.cat(ys2, dim=1)
        
    diff_y2 = (y_chunkwise2 - y_stepwise2).abs().max().item()
    diff_state2 = (state_chunkwise2 - curr_state2).abs().max().item()
    print(f"  - Output Max Diff: {diff_y2:.6e}")
    print(f"  - State  Max Diff: {diff_state2:.6e}")
    
    assert torch.allclose(y_chunkwise2, y_stepwise2, atol=1e-4, rtol=1e-4), f"Outputs mismatch: {diff_y2}"
    assert torch.allclose(state_chunkwise2, curr_state2, atol=1e-4, rtol=1e-4), f"States mismatch: {diff_state2}"
    print("✓ Test 2 Passed (with initial_state)!")

def test_full_hope_model():
    print("\nTesting full HOPE model stateful forward...")
    config = HOPEConfig(n_embd=256, n_head=4, n_layer=2, block_size=256)
    model = HOPE(config).eval()
    
    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    
    with torch.no_grad():
        logits, loss, states = model(idx)
        assert len(states) == config.n_layer
        assert states[0].shape == (B, config.n_head, config.n_embd // config.n_head, config.n_embd // config.n_head)
        
        # Second forward pass with persistent state
        logits2, loss2, states2 = model(idx, states=states)
        assert len(states2) == config.n_layer
    print("✓ Full HOPE model test passed!")

if __name__ == "__main__":
    test_titans_equivalence()
    test_full_hope_model()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY! 🎉")
