# minigpt_m1_safe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# M1-friendly Hyperparameters (conservative)
# -------------------------
vocab_size = 100
d_model = 256       # smaller embedding size (divisible by n_heads)
n_heads = 16         # head_dim = 12
n_layers = 8
d_ff = d_model * 4
seq_len = 30        # shorter sequence for lower memory footprint
batch_size = 1      # single example at a time
dropout = 0.05

# Device selection (M1-aware)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Helpful safety flags
USE_CHECKPOINT = False      # disable checkpoint by default on MPS
RETURN_WEIGHTS = False      # do not return attention weights by default (saves memory)
PRINT_SHAPES = True

# -------------------------
# Modules (same logic)
# -------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
    def forward(self, tokens:torch.Tensor):
        return self.emb(tokens)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len:int, d_model:int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
    def forward(self, x:torch.Tensor):
        batch, seq, _ = x.shape
        positions = torch.arange(seq, device=x.device).unsqueeze(0).expand(batch, seq)
        return self.pos_emb(positions)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, causal:bool=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, need_weights:bool=False):
        b, seq, _ = x.shape

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        def reshape_head(t):
            return t.view(b, seq, self.n_heads, self.head_dim).transpose(1,2)

        Qh = reshape_head(Q)
        Kh = reshape_head(K)
        Vh = reshape_head(V)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if self.causal:
            idxs = torch.arange(seq, device=x.device)
            mask = idxs.unsqueeze(0) <= idxs.unsqueeze(1)   # (seq, seq)
            mask = mask.unsqueeze(0).unsqueeze(0)           # (1,1,seq,seq)
            # MPS-friendly large negative number (avoid -inf)
            scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out_heads = torch.matmul(attn, Vh)
        out = out_heads.transpose(1,2).contiguous().view(b, seq, self.d_model)
        out = self.out(out)

        if need_weights:
            return out, attn
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x:torch.Tensor):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float=0.05, use_checkpoint:bool=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, causal=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def forward(self, x:torch.Tensor, need_weights:bool=False):
        # Note: checkpointing typical usage wraps the function that computes the block.
        x_norm = self.ln1(x)
        if need_weights:
            attn_out, attn_weights = self.attn(x_norm, need_weights=True)
        else:
            attn_out = self.attn(x_norm, need_weights=False)
            attn_weights = None

        x = x + self.dropout(attn_out)
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        return (x, attn_weights) if need_weights else x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size:int, max_len:int, d_model:int, n_heads:int, 
                 n_layers:int, d_ff:int, dropout:float=0.05, use_checkpoint:bool=False):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_checkpoint) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens:torch.Tensor, need_weights:bool=False):
        x = self.tok_emb(tokens)
        x = x + self.pos_emb(x)
        x = self.drop(x)

        attn_weights_all = [] if need_weights else None
        for block in self.blocks:
            if need_weights:
                x, attn = block(x, need_weights=True)
                attn_weights_all.append(attn)
            else:
                x = block(x, need_weights=False)

        x = self.ln_f(x)
        logits = self.head(x)
        return (logits, attn_weights_all) if need_weights else logits

# -------------------------
# Run a safe forward pass
# -------------------------
if __name__ == "__main__":
    model = MiniGPT(
        vocab_size=vocab_size, 
        max_len=seq_len, 
        d_model=d_model, 
        n_heads=n_heads, 
        n_layers=n_layers, 
        d_ff=d_ff,
        dropout=dropout,
        use_checkpoint=USE_CHECKPOINT
    ).to(device)

    model.eval()

    # Create tokens directly on the device to avoid extra copies
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    if PRINT_SHAPES:
        print("Model params (conservative M1 settings):")
        print(f" batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        print(f"Using device: {device}")
        print("Dummy tokens shape:", tokens.shape)

    # Do NOT request attention weights unless you need them
    with torch.no_grad():
        logits = model(tokens, need_weights=RETURN_WEIGHTS)

    # with torch.no_grad():
    #     logits, attn_weights = model(tokens, need_weights=True)
    #     print("Attn shape (block0):", attn_weights[0].shape)  # expect (1, 4, 8, 8)
    #     print(attn_weights[0][0,0])  # head 0, batch 0


    if PRINT_SHAPES:
        print("Logits shape:", logits.shape if not RETURN_WEIGHTS else logits[0].shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
