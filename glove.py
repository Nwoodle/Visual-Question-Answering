import numpy as np

dim = 100

word2emb = {}

with open(f"glove.6B.{dim}d.txt", 'r', encoding='utf-8') as f:
    entries = f.readlines()

emb_dim = len(entries[0].split(' ')) - 1


for entry in entries:

    vals = entry.split(' ')
    word = vals[0]
    vals = vals[1:]
    word2emb[word] = np.array(vals).astype(np.float)

def word2embedding(word):
    try:
        return word2emb[word]
    except:
        return np.zeros(dim)

        