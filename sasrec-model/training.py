import numpy as np
import torch
import torch.nn as nn
import pickle
import time
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Training Setup
BATCH_SIZE = 512
LR = 0.001
MAX_LEN = 50
HIDDEN_UNITS = 128
NUM_BLOCKS = 2 # Transformer layers
NUM_HEADS = 2 # Attention heads
DROPOUT = 0.2
EPOCHS = 20
DEVICE = torch.device('cuda')

class GoodreadsDataset(Dataset):
    def __init__(self, user_train, item_num, max_len):
        self.user_train = user_train
        self.item_num = item_num
        self.max_len = max_len
        self.users = list(user_train.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_train[user]

        # Truncate to last max_len
        seq = seq[-self.max_len:]

        # Set input and target
        seq_input = seq[:-1]
        pos_target = seq[1:]

        # Negative Sampling
        neg_target = []
        for _ in range(len(pos_target)):
            neg = np.random.randint(1, self.item_num + 1)
            while neg in seq: # collision check
                neg = np.random.randint(1, self.item_num + 1)
            neg_target.append(neg)

        # Padding
        pad_len = self.max_len - len(seq_input)
        if pad_len > 0:
            seq_input = [0] * pad_len + seq_input
            pos_target = [0] * pad_len + pos_target
            neg_target = [0] * pad_len + neg_target
        
        return torch.LongTensor(seq_input), torch.LongTensor(pos_target), torch.LongTensor(neg_target)

class SASRec(nn.Module):
    def __init__(self, item_num, hidden_units, max_len, num_blocks, num_heads, dropout):
        super(SASRec, self).__init__()
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=hidden_units*4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(num_blocks)
        ])

        self.last_norm = nn.LayerNorm(hidden_units)

    def forward(self, seqs, pos_items, neg_items):
        seqs_emb = self.item_emb(seqs)
        positions = torch.arange(seqs.shape[1], device=seqs.device).unsqueeze(0)
        seqs_emb += self.pos_emb(positions)
        seqs_emb = self.emb_dropout(seqs_emb)

        L = seqs.shape[1]
        mask = torch.triu(torch.ones(L, L, device=seqs.device) * float('-inf'), diagonal=1)

        key_padding_mask = (seqs == 0)

        for block in self.blocks:
            seqs_emb = block(seqs_emb, src_mask=mask, src_key_padding_mask=key_padding_mask)
        
        seqs_emb = self.last_norm(seqs_emb)

        pos_emb = self.item_emb(pos_items)
        neg_emb = self.item_emb(neg_items)

        pos_logits = (seqs_emb * pos_emb).sum(dim=-1)
        neg_logits = (seqs_emb * neg_emb).sum(dim=-1)

        return pos_logits, neg_logits

def main():
    print("Loading dataset...")
    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    dataset = GoodreadsDataset(data['user_train'], data['item_num'], MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = SASRec(data['item_num'], HIDDEN_UNITS, MAX_LEN, NUM_BLOCKS, NUM_HEADS, DROPOUT).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    scaler = GradScaler("cuda")

    print("TRAINING IN PROGRESS...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()
        
        for step, (seqs, pos, neg) in enumerate(dataloader):
            seqs, pos, neg = seqs.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast("cuda"):
                pos_logits, neg_logits = model(seqs, pos, neg)
                
                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)
                
                loss = criterion(pos_logits, pos_labels) + criterion(neg_logits, neg_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if step % 100 == 0 and step > 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} done. Avg Loss: {total_loss / len(dataloader):.4f}. Time: {time.time() - start_time:.1f}s")
        
        torch.save(model.state_dict(), f"sasrec_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    main()