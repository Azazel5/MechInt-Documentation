import torch


torch.manual_seed(1337)

def encode(text, mapping):
    return [mapping[c] for c in text]

def decode(text, mapping):
    return ''.join([mapping[i] for i in text])

def get_batch(split, train_data, val_data, block_size, batch_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def main():
    with open("input.txt", "r") as f:
        data = f.read()

    chars = sorted(list(set(data)))

    # Mapping dictionaries for encoding and decoding
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    data_torch = torch.tensor(encode(data, stoi), dtype=torch.long) 
    print(f"Data has {len(data_torch)} characters, {len(chars)} unique.")
    
    # Splitting into train and validation sets
    n = int(0.9*len(data_torch))
    train_data = data_torch[:n]
    val_data = data_torch[n:]

    batch_size = 4 # how many independent sequences will we process in parallel?
    block_size = 8 # what is the maximum context length for predictions?

    xb, yb = get_batch('train', train_data, val_data, block_size, batch_size)
    print('inputs:')
    print(xb.shape)
    print(xb)
    print('targets:')
    print(yb.shape)
    print(yb)

    print('----')

    for b in range(batch_size): # batch dimension
        for t in range(block_size): # time dimension
            context = xb[b, :t+1]
            target = yb[b,t]
            context_text = decode(context.tolist(), itos)
            target_char = decode([target.item()], itos)
            # print(f"when input is {context.tolist()} ({context_text!r}) the target: {target.item()} ({target_char!r})")

    

if __name__ == "__main__":
    main()