"""
HOPE Model Reference Implementation (High-Fidelity)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from config import HOPEConfig

class TitansL2(nn.Module):
    """
    Titans Memory Module with L2/Delta Rule Update.
    
    Implements the update rule:
    M_{t+1} = M_t (I - alpha * k_t k_t^T) + beta * v_t k_t^T
    
    This allows the memory to "forget" information along direction k_t before writing v_t.
    
    Optimization: Chunkwise Parallel Scan
    - Splits sequence into chunks.
    - Computes chunk-level transition matrices (Decay A, Update B) in parallel.
    - Runs a global scan over chunks.
    - Computes intra-chunk outputs in parallel.
    """
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.chunk_size = 128 # Tunable chunk size
        
        # Projections
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Learnable parameters (bounded to prevent explosion)
        # Raw parameters that will be passed through sigmoid
        self.alpha_raw = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        self.beta_raw = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
    
    @property
    def alpha(self):
        # Bound alpha to [0, 0.5] to prevent memory collapse
        return torch.sigmoid(self.alpha_raw) * 0.5
    
    @property
    def beta(self):
        # Bound beta to [0, 0.5] for stable writes
        return torch.sigmoid(self.beta_raw) * 0.5

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, H, T, D)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Normalize keys
        k = F.normalize(k, dim=-1)
        
        if state is not None:
            return self.forward_inference(q, k, v, state)
        else:
            return self.forward_train_chunkwise(q, k, v)

    def forward_inference(self, q, k, v, state):
        # q, k, v: (B, H, 1, D)
        # state: (B, H, D, D)
        
        # 1. Read: y = q @ M^T
        y = torch.matmul(q, state.transpose(-1, -2)) 
        
        # 2. Update
        k_t = k.transpose(-1, -2) # (B, H, D, 1)
        v_t = v.transpose(-1, -2) # (B, H, D, 1)
        
        # M_new = M - alpha * (M k) k^T + beta * v k^T
        Mk = torch.matmul(state, k_t) # (B, H, D, 1)
        forget_term = torch.matmul(Mk, k) # (B, H, D, D)
        write_term = torch.matmul(v_t, k) # (B, H, D, D)
        
        new_state = state - self.alpha * forget_term + self.beta * write_term
        
        # Output projection
        B, H, T, D = y.shape
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.c_proj(y)
        
        return y, new_state

    def forward_train_chunkwise(self, q, k, v):
        """
        Chunkwise Parallel Scan Implementation.
        """
        B, H, T, D = q.shape
        chunk_size = self.chunk_size
        
        # Pad if necessary
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            T_padded = T + pad_len
        else:
            T_padded = T
            
        num_chunks = T_padded // chunk_size
        
        # Reshape to chunks: (B, H, num_chunks, chunk_size, D)
        q_chunks = q.view(B, H, num_chunks, chunk_size, D)
        k_chunks = k.view(B, H, num_chunks, chunk_size, D)
        v_chunks = v.view(B, H, num_chunks, chunk_size, D)
        
        # ---------------------------------------------------------------------
        # Step 1: Compute Chunk Operators (A_chunk, B_chunk) in Parallel
        # ---------------------------------------------------------------------
        # We need to compute the cumulative effect of a chunk on the memory state.
        # M_{out} = M_{in} @ A + B
        # This is expensive to do purely in parallel (D^3), so we use a compiled 
        # sequential loop over the small chunk_size.
        
        # We'll use a helper function that can be torch.compiled
        A_chunks, B_chunks = self._compute_chunk_operators(k_chunks, v_chunks)
        
        # ---------------------------------------------------------------------
        # Step 2: Global Scan over Chunks
        # ---------------------------------------------------------------------
        # Compute M_{start} for each chunk using the operators.
        # M_0 = 0
        # M_1 = M_0 A_0 + B_0
        # M_2 = M_1 A_1 + B_1 ...
        
        M_starts = [torch.zeros(B, H, D, D, device=q.device)]
        curr_M = M_starts[0]
        
        # This loop runs over num_chunks (e.g. 1024/128 = 8 steps), so it's fast.
        for i in range(num_chunks):
            # A: (B, H, D, D), B: (B, H, D, D)
            A = A_chunks[:, :, i]
            B_op = B_chunks[:, :, i]
            
            # M_{i+1} = M_i @ A + B
            next_M = torch.matmul(curr_M, A) + B_op
            M_starts.append(next_M)
            curr_M = next_M
            
        # Stack M_starts: (B, H, num_chunks, D, D)
        # We only need M_starts[0...num_chunks-1] for processing chunks
        M_starts_tensor = torch.stack(M_starts[:-1], dim=2)
        
        # ---------------------------------------------------------------------
        # Step 3: Intra-Chunk Processing (Parallel)
        # ---------------------------------------------------------------------
        # Now we have the correct initial state for each chunk.
        # We can process all chunks in parallel to get outputs y.
        
        y_chunks = self._process_chunks(q_chunks, k_chunks, v_chunks, M_starts_tensor)
        
        # Reshape back
        y = y_chunks.view(B, H, T_padded, D)
        if T != T_padded:
            y = y[:, :, :T, :]
            
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        
        # Return final state (M_starts[-1] is the state after the last chunk)
        return self.c_proj(y), M_starts[-1]

    def _compute_chunk_operators(self, k_chunks, v_chunks):
        """
        Computes A (decay) and B (update) matrices for each chunk.
        M_{out} = M_{in} A + B
        """
        B, H, num_chunks, chunk_size, D = k_chunks.shape
        
        # We will iterate sequentially over chunk_size, but parallel over B, H, num_chunks
        # Flatten batch dims for easier processing
        # (Batch_Total, chunk_size, D) where Batch_Total = B * H * num_chunks
        k_flat = k_chunks.reshape(-1, chunk_size, D)
        v_flat = v_chunks.reshape(-1, chunk_size, D)
        
        # Initialize operators
        # A starts as Identity
        A = torch.eye(D, device=k_chunks.device).unsqueeze(0).expand(k_flat.size(0), D, D).clone()
        # B starts as Zero
        B_op = torch.zeros_like(A)
        
        alpha = self.alpha.view(1, H, 1, 1).expand(B, H, num_chunks, 1).reshape(-1, 1, 1)
        beta = self.beta.view(1, H, 1, 1).expand(B, H, num_chunks, 1).reshape(-1, 1, 1)
        
        # Loop over chunk_size (e.g., 128)
        for t in range(chunk_size):
            kt = k_flat[:, t, :].unsqueeze(2) # (BT, D, 1)
            vt = v_flat[:, t, :].unsqueeze(2) # (BT, D, 1)
            
            # Update Rule: M_{t} = M_{t-1} (I - alpha k k^T) + beta v k^T
            # Let Step Operator S_t(M) = M D_t + U_t
            # D_t = I - alpha k k^T
            # U_t = beta v k^T
            
            # Composition: M_t = (M_{t-1} A_{t-1} + B_{t-1}) D_t + U_t
            #                  = M_{t-1} (A_{t-1} D_t) + (B_{t-1} D_t + U_t)
            # So: A_new = A_old @ D_t
            #     B_new = B_old @ D_t + U_t
            
            kt_T = kt.transpose(1, 2)
            
            # D_t = I - alpha * k * k^T
            # We can compute A @ D_t directly:
            # A @ (I - alpha k k^T) = A - alpha (A k) k^T
            Ak = torch.matmul(A, kt) # (BT, D, 1)
            A = A - alpha * torch.matmul(Ak, kt_T)
            
            # B @ D_t + U_t
            # B - alpha (B k) k^T + beta v k^T
            Bk = torch.matmul(B_op, kt)
            B_op = B_op - alpha * torch.matmul(Bk, kt_T) + beta * torch.matmul(vt, kt_T)
            
        # Reshape back
        A = A.view(B, H, num_chunks, D, D)
        B_op = B_op.view(B, H, num_chunks, D, D)
        
        return A, B_op

    def _process_chunks(self, q_chunks, k_chunks, v_chunks, M_starts):
        """
        Computes outputs y within chunks given initial states M_starts.
        """
        B, H, num_chunks, chunk_size, D = q_chunks.shape
        
        # Flatten
        q_flat = q_chunks.reshape(-1, chunk_size, D)
        k_flat = k_chunks.reshape(-1, chunk_size, D)
        v_flat = v_chunks.reshape(-1, chunk_size, D)
        M_curr = M_starts.reshape(-1, D, D).clone()
        
        alpha = self.alpha.view(1, H, 1, 1).expand(B, H, num_chunks, 1).reshape(-1, 1, 1)
        beta = self.beta.view(1, H, 1, 1).expand(B, H, num_chunks, 1).reshape(-1, 1, 1)
        
        ys = []
        
        for t in range(chunk_size):
            qt = q_flat[:, t, :].unsqueeze(2)
            kt = k_flat[:, t, :].unsqueeze(2)
            vt = v_flat[:, t, :].unsqueeze(2)
            
            # Read: y = q @ M^T
            yt = torch.matmul(qt, M_curr.transpose(1, 2))
            ys.append(yt)
            
            # Update M
            kt_T = kt.transpose(1, 2)
            Mk = torch.matmul(M_curr, kt)
            
            forget = torch.matmul(Mk, kt_T)
            write = torch.matmul(vt, kt_T)
            
            M_curr = M_curr - alpha * forget + beta * write
            
        y = torch.cat(ys, dim=2) # (BT, chunk_size, D)
        y = y.view(B, H, num_chunks, chunk_size, D)
        return y

class CMSBlock(nn.Module):
    """
    Continuum Memory System Block.
    Standard MLP but designed to be part of the multi-rate hierarchy.
    """
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class HOPEBlock(nn.Module):
    def __init__(self, config: HOPEConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.titans = TitansL2(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.cms = CMSBlock(config)

    def forward(self, x, state: Optional[torch.Tensor] = None):
        # Titans Part
        # state passed here is specifically for this layer's Titans module
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

    def forward(self, idx, targets=None, states=None, pos_offset=0):
        """
        Args:
            idx: input token indices (B, T)
            targets: target token indices for loss computation
            states: list of memory states from previous forward pass
            pos_offset: position offset for stateful generation (critical for correct positional encoding)
        """
        device = idx.device
        b, t = idx.size()
        
        # CRITICAL FIX: Use pos_offset for stateful generation
        # During generation, pos_offset tracks the true position in the sequence
        pos = torch.arange(pos_offset, pos_offset + t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        new_states = []
        
        for i, block in enumerate(self.transformer.h):
            # Get state for this block if available
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
        
        # Take the last token's logits
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Yield the prompt (optional, but here we just append)
        out = torch.cat((idx, idx_next), dim=1)
        
        # 2. Generation Loop: O(1) per token with CORRECT positions
        current_pos = idx.size(1)  # Track the true position
        
        for _ in range(max_new_tokens - 1):
            # CRITICAL FIX: Pass pos_offset so the model knows the true position
            logits, _, states = self(idx_next, states=states, pos_offset=current_pos)
            
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            out = torch.cat((out, idx_next), dim=1)
            current_pos += 1  # CRITICAL: Increment position counter
            
        return out

