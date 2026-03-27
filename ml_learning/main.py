import torch
import torch.nn as nn

with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

str_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_str = { i:ch for i,ch in enumerate(chars) }

tokens = torch.tensor([str_to_int[ch] for ch in text])


print(f"Vocab size: {vocab_size}")
print(f"Total Tokens: {len(tokens)}")

embedding_dim = 64
num_heads     = 4
num_layers    = 4
block_size = 64
batch_size = 16

class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_size):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key   = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # computing attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (K.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)

        # blending the vectors to using attention weights
        return weights @ V
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([
            AttentionHead(embedding_dim, self.head_size)
            for _ in range(num_heads)
        ])
        # creating a weight matrix to combine head outputs
        self.proj = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x): 
        out = torch.cat([head(x) for head in self.heads], dim = -1)

        # combing head outputs using the weight matrix (how much does each head effect the vector values?)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens):
        x = self.embedding(tokens)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.output(x)
        return x

model = TinyGPT(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
)

def generate(model, tokens, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(tokens)
        last_logits = logits[-1]
        probs = torch.softmax(last_logits, dim =-1)
        next_token = torch.multinomial(probs, num_samples = 1)
        tokens = torch.cat([tokens, next_token])
    return tokens

def decode(integers):
    return "".join([int_to_str[i] for i in integers])
# adjusting weights during training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

def get_batch(tokens):
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x  = torch.stack([tokens[i:i+block_size] for i in ix])
    y  = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x, y
# training loop
for step in range(10000):

    x, y = get_batch(tokens)
    logits = model(x)

    # calculating loss - how wrong is the model? 
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), # all predictions except the last
        y.view(-1)   # all tokens except the first
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")

user_input = input("Write something: ")
seed = torch.tensor([str_to_int[ch] for ch in user_input])
output = generate(model, seed, max_new_tokens=200)
print(decode(output.tolist()))