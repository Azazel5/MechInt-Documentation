import torch
from model import BigramLanguageModel


torch.manual_seed(1337)

batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def encode(text, mapping):
    return [mapping[c] for c in text]

def decode(text, mapping):
    return ''.join([mapping[i] for i in text])

def get_batch(split, train_data, val_data):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, train_data, val_data)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    with open("input.txt", "r") as f:
        data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    # Mapping dictionaries for encoding and decoding
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    data_torch = torch.tensor(encode(data, stoi), dtype=torch.long)
    print(f"Data has {len(data_torch)} characters, {len(chars)} unique.")

    # Splitting into train and validation sets
    n = int(0.9*len(data_torch))
    train_data = data_torch[:n]
    val_data = data_torch[n:]

    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())
    seed_text = "Here we go"
    context = torch.tensor([encode(seed_text, stoi)], dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=300)[0].tolist(), itos))

if __name__ == "__main__":
    main()
