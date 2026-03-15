import torch
import torch.nn as nn
from torch.nn import functional as F
from spacy.lang.ru import Russian

# hypr params
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 64  # what is the maximum context length for predictions?
max_iter = 2000
eval_iters = 200
eta = 3e-4
n_emb = 256
n_head = 10
n_layer = 10
dropout_n = 0.35
data_split = 0.8
device = 'cuda'
word_token = True

# --------------------
torch.manual_seed(1337)
# src_filename = 'Data/Datasets/Infinite sadness.txt'
src_filename = 'Data/Datasets/LLM_dataset_2_daniel_plain.txt'

with open(src_filename, 'r') as f:
    data = f.read()

# Word level tokens
if word_token:
    nlp = Russian()


    def get_word_tokens(text):
        doc = nlp(text)
        return [token.text for token in doc]


    def encode(text):
        doc = nlp(text)
        tok_text = (token.text for token in doc)
        return [stoi[c] for c in tok_text]


    def decode(enc_text):
        return ' '.join([itos[c] for c in enc_text])


    data = data.lower()
    vocab = sorted(set(get_word_tokens(data)))
    vocab_size = len(vocab)

    # encoding chars to ints and vise versa
    itos = {key: tocken for key, tocken in enumerate(vocab)}
    stoi = {tocken: key for key, tocken in enumerate(vocab)}

# letter level tokens
else:
    # getting channels
    vocab = sorted(set(data))
    vocab_size = len(vocab)

    # encoding chars to ints and vise versa
    itos = {key: tocken for key, tocken in enumerate(vocab)}
    stoi = {tocken: key for key, tocken in enumerate(vocab)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda i: ''.join([itos[c] for c in i])

# splitting dataset
data = torch.tensor(encode(data), dtype=torch.long)  # torch.Size([345885])
n = int(data_split * len(data))
train_x = data[:n]
val_x = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_x if split == 'train' else val_x
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


xb, yb = get_batch('train')


@torch.no_grad() # don't save gradients as we're not moving backwards
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Head object of self attention"""

    def __init__(self, head_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout_n)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1)  # (B, T, C) @ (B, C, T) ----> (B, T, T)
        # wei = torch.zeros((T, T))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.drop(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ multiple Head attentions in parallel """

    def __init__(self, num_heads, head_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.drop = nn.Dropout(dropout_n)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.drop(out)

        return out


class FeedForward(nn.Module):
    """ linear layer followed by non-linearity """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout_n),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ transformer block: communication followed by computation """
    def __init__(self, n_head):
        # n_emb: embedding dim, n_head: number of heads
        super().__init__()
        head_size = n_emb // n_head  # if n_emb = 32, head_size must be 8, therefore n_head = 4
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffd = FeedForward()
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = self.sa(self.ln1(x))
        x = self.ffd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.pos_emb_table = nn.Embedding(block_size, n_emb)
        # self.sa_head = Head(n_emb)  # self attention head
        self.sa_head = MultiHeadAttention(4, n_emb // 4)  # id est 4 heads of 8-dim self-attention (convolution)
        # self.ffd = FeedForward()
        self.blocks = nn.Sequential(*[Block(n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,C) = (batch=4, time=8, channel=169)
        pos_emb = self.pos_emb_table(torch.arange(T, device=device))  # (T, C)
        x = pos_emb + tok_emb  # (B, T, C)
        x = self.sa_head(x)  # apply one head of self-attention (B,T,C)
        # x = self.ffd(x)  # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)  # (-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current contex
        for _ in range(max_new_tokens):
            # crop  idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)  # calls forward method
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), eta)

for iter in range(max_iter):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

PATH = 'Data/Weights/Infinite sadness.pt'
torch.save(model.state_dict(), PATH)

# generate from model
contex = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(contex, max_new_tokens=10000)[0].tolist()))
