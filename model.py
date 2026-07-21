"""
HOPE Model Implementation
Self-Modifying Titans Core + Continuum Memory System (CMS)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from config import HOPEConfig

class TitansL2(nn.Module):
    """
    Titans Memory Module with L2/Delta Rule Update and Data-Dependent Decay/Write.
    
    Implements the update rule:
    M_{t+1} = M_t (I - alpha_t k_t k_t^T) + beta_t v_t k_t^T
    
    where alpha_t and beta_t are per-token data-dependent gating scalars:
    alpha_t = sigmoid(Linear_alpha(x_t)) * 0.5
    beta_t  = sigmoid(Linear_beta(x_t)) * 0.5
    """
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.chunk_size = 32  # Tunable chunk size (smaller = less CUDA memory)
        
        # Projections for Q, K, V
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Data-dependent alpha and beta projections per token and per head
        self.c_alpha = nn.Linear(config.n_embd, config.n_head, bias=config.bias)
        self.c_beta = nn.Linear(config.n_embd, config.n_head, bias=config.bias)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, H, T, D)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, H, T, D)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, H, T, D)
        
        # Normalize keys
        k = F.normalize(k, dim=-1)
        
        # Compute per-token data-dependent alpha and beta: shape (B, T, H)
        alpha = torch.sigmoid(self.c_alpha(x)) * 0.5
        beta = torch.sigmoid(self.c_beta(x)) * 0.5
        
        if T == 1:
            if state is None:
                state = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
            return self.forward_inference(q, k, v, alpha, beta, state)
        else:
            return self.forward_train_chunkwise(q, k, v, alpha, beta, state=state)

    def forward_inference(self, q, k, v, alpha, beta, state):
        """
        True single-token (T=1) autoregressive decoding step.
        q, k, v: (B, H, 1, D)
        alpha, beta: (B, 1, H)
        state: (B, H, D, D)
        """
        # 1. Read: y = q @ M^T
        y = torch.matmul(q, state.transpose(-1, -2)) # (B, H, 1, D)
        
        # 2. Update
        k_t = k.transpose(-1, -2) # (B, H, D, 1)
        v_t = v.transpose(-1, -2) # (B, H, D, 1)
        
        # Reshape alpha and beta to (B, H, 1, 1)
        alpha_h = alpha.transpose(1, 2).unsqueeze(-1) # (B, H, 1, 1)
        beta_h = beta.transpose(1, 2).unsqueeze(-1)   # (B, H, 1, 1)
        
        # M_new = M - alpha * (M k) k^T + beta * v k^T
        Mk = torch.matmul(state, k_t) # (B, H, D, 1)
        forget_term = torch.matmul(Mk, k) # (B, H, D, D)
        write_term = torch.matmul(v_t, k) # (B, H, D, D)
        
        new_state = state - alpha_h * forget_term + beta_h * write_term
        
        # Output projection
        B, H, T, D = y.shape
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.c_proj(y)
        
        return y, new_state

    def forward_train_chunkwise(self, q, k, v, alpha, beta, state: Optional[torch.Tensor] = None):
        """
        Chunkwise Parallel Scan Implementation supporting initial state M_starts[0].
        q, k, v: (B, H, T, D)
        alpha, beta: (B, T, H)
        state: optional initial memory state (B, H, D, D)
        """
        B, H, T, D = q.shape
        chunk_size = self.chunk_size
        
        # Pad if necessary
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            alpha_pad = F.pad(alpha, (0, 0, 0, pad_len))
            beta_pad = F.pad(beta, (0, 0, 0, pad_len))
            T_padded = T + pad_len
        else:
            alpha_pad = alpha
            beta_pad = beta
            T_padded = T
            
        num_chunks = T_padded // chunk_size
        
        # Reshape to chunks: (B, H, num_chunks, chunk_size, D)
        q_chunks = q.view(B, H, num_chunks, chunk_size, D)
        k_chunks = k.view(B, H, num_chunks, chunk_size, D)
        v_chunks = v.view(B, H, num_chunks, chunk_size, D)
        
        # alpha_pad: (B, T_padded, H) -> transpose to (B, H, T_padded) -> view (B, H, num_chunks, chunk_size)
        alpha_chunks = alpha_pad.transpose(1, 2).view(B, H, num_chunks, chunk_size)
        beta_chunks = beta_pad.transpose(1, 2).view(B, H, num_chunks, chunk_size)
        
        # ---------------------------------------------------------------------
        # Step 1: Compute Chunk Operators (A_chunk, B_chunk) in Parallel
        # ---------------------------------------------------------------------
        A_chunks, B_chunks = self._compute_chunk_operators(k_chunks, v_chunks, alpha_chunks, beta_chunks)
        
        # ---------------------------------------------------------------------
        # Step 2: Global Scan over Chunks
        # ---------------------------------------------------------------------
        if state is not None:
            M_starts = [state]
        else:
            M_starts = [torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)]
            
        curr_M = M_starts[0]
        
        for i in range(num_chunks):
            A = A_chunks[:, :, i]
            B_op = B_chunks[:, :, i]
            
            # M_{i+1} = M_i @ A + B
            next_M = torch.matmul(curr_M, A) + B_op
            M_starts.append(next_M)
            curr_M = next_M
            
        # Stack M_starts: (B, H, num_chunks, D, D)
        M_starts_tensor = torch.stack(M_starts[:-1], dim=2)
        
        # ---------------------------------------------------------------------
        # Step 3: Intra-Chunk Processing (Parallel)
        # ---------------------------------------------------------------------
        y_chunks = self._process_chunks(q_chunks, k_chunks, v_chunks, alpha_chunks, beta_chunks, M_starts_tensor)
        
        # Reshape back
        y = y_chunks.view(B, H, T_padded, D)
        if T != T_padded:
            y = y[:, :, :T, :]
            
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        
        return self.c_proj(y), M_starts[-1]

    def _compute_chunk_operators(self, k_chunks, v_chunks, alpha_chunks, beta_chunks):
        """
        Computes A (decay) and B (update) matrices for each chunk.
        M_{out} = M_{in} A + B
        """
        B, H, num_chunks, chunk_size, D = k_chunks.shape
        
        k_flat = k_chunks.reshape(-1, chunk_size, D)
        v_flat = v_chunks.reshape(-1, chunk_size, D)
        alpha_flat = alpha_chunks.reshape(-1, chunk_size, 1)
        beta_flat = beta_chunks.reshape(-1, chunk_size, 1)
        
        A = torch.eye(D, device=k_chunks.device, dtype=k_chunks.dtype).unsqueeze(0).expand(k_flat.size(0), D, D).clone()
        B_op = torch.zeros_like(A)
        
        for t in range(chunk_size):
            kt = k_flat[:, t, :].unsqueeze(2) # (BT, D, 1)
            vt = v_flat[:, t, :].unsqueeze(2) # (BT, D, 1)
            alpha_t = alpha_flat[:, t, :].unsqueeze(2) # (BT, 1, 1)
            beta_t = beta_flat[:, t, :].unsqueeze(2)   # (BT, 1, 1)
            
            kt_T = kt.transpose(1, 2)
            
            Ak = torch.matmul(A, kt) # (BT, D, 1)
            A = A - alpha_t * torch.matmul(Ak, kt_T)
            
            Bk = torch.matmul(B_op, kt)
            B_op = B_op - alpha_t * torch.matmul(Bk, kt_T) + beta_t * torch.matmul(vt, kt_T)
            
        A = A.view(B, H, num_chunks, D, D)
        B_op = B_op.view(B, H, num_chunks, D, D)
        
        return A, B_op

    def _process_chunks(self, q_chunks, k_chunks, v_chunks, alpha_chunks, beta_chunks, M_starts):
        """
        Computes outputs y within chunks given initial states M_starts.
        """
        B, H, num_chunks, chunk_size, D = q_chunks.shape
        
        q_flat = q_chunks.reshape(-1, chunk_size, D)
        k_flat = k_chunks.reshape(-1, chunk_size, D)
        v_flat = v_chunks.reshape(-1, chunk_size, D)
        alpha_flat = alpha_chunks.reshape(-1, chunk_size, 1)
        beta_flat = beta_chunks.reshape(-1, chunk_size, 1)
        M_curr = M_starts.reshape(-1, D, D).clone()
        
        ys = []
        
        for t in range(chunk_size):
            qt = q_flat[:, t, :].unsqueeze(1) # (BT, 1, D)
            kt = k_flat[:, t, :].unsqueeze(2) # (BT, D, 1)
            vt = v_flat[:, t, :].unsqueeze(2) # (BT, D, 1)
            alpha_t = alpha_flat[:, t, :].unsqueeze(2) # (BT, 1, 1)
            beta_t = beta_flat[:, t, :].unsqueeze(2)   # (BT, 1, 1)
            
            # Read: y = q @ M^T
            yt = torch.matmul(qt, M_curr.transpose(1, 2)) # (BT, 1, D)
            ys.append(yt)
            
            # Update M
            kt_T = kt.transpose(1, 2)
            Mk = torch.matmul(M_curr, kt)
            
            forget = torch.matmul(Mk, kt_T)
            write = torch.matmul(vt, kt_T)
            
            M_curr = M_curr - alpha_t * forget + beta_t * write
            
        y = torch.cat(ys, dim=1) # (BT, chunk_size, D)
        y = y.view(B, H, num_chunks, chunk_size, D)
        return y

class CMSBlock(nn.Module):
    """
    Continuum Memory System Block.
    Standard MLP operating with multi-rate parameter updates determined by config.cms_update_periods.
    """
    def __init__(self, config: HOPEConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        periods = config.cms_update_periods
        self.period = periods[layer_idx % len(periods)] if periods else 1
        
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

    def enforce_update_period(self, step: int):
        """Zero out gradients if current step is not an update step for this layer's period."""
        if self.period > 1 and step % self.period != 0:
            for p in self.parameters():
                p.grad = None

class HOPEBlock(nn.Module):
    def __init__(self, config: HOPEConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.titans = TitansL2(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.cms = CMSBlock(config, layer_idx=layer_idx)

    def forward(self, x, state: Optional[torch.Tensor] = None):
        # Titans Part
        res, new_state = self.titans(self.ln1(x), state)
        x = x + res
        
        # CMS Part
        x = x + self.cms(self.ln2(x))
        
        return x, new_state

class HOPE(nn.Module):
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([HOPEBlock(config, i) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enforce_cms_update_periods(self, step: int):
        """Enforces multi-rate updates on CMS blocks."""
        for block in self.transformer.h:
            block.cms.enforce_update_period(step)

    def forward(self, idx, targets=None, states=None, pos_offset=0):
        """
        Args:
            idx: input token indices (B, T)
            targets: target token indices for loss computation
            states: list of memory states from previous forward pass
            pos_offset: position offset for stateful generation
        """
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(pos_offset, pos_offset + t, dtype=torch.long, device=device) % self.config.block_size
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        new_states = []
        
        for i, block in enumerate(self.transformer.h):
            block_state = states[i] if states is not None else None
            x, new_block_state = block(x, state=block_state)
            new_states.append(new_block_state)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss, new_states

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Stateful generation with correct positional encoding.
        """
        # 1. Prefill: process the prompt
        logits, _, states = self(idx, pos_offset=0)
        
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        out = torch.cat((idx, idx_next), dim=1)
        
        # 2. Generation Loop: O(1) per token with CORRECT positions
        current_pos = idx.size(1)
        
        for _ in range(max_new_tokens - 1):
            logits, _, states = self(idx_next, states=states, pos_offset=current_pos)
            
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            out = torch.cat((out, idx_next), dim=1)
            current_pos += 1
            
        return out
