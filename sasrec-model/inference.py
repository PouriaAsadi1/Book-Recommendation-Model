import torch
import torch.nn as nn
import pickle
import numpy as np
import sys

# Inferencing Setup
MAX_LEN = 50
HIDDEN_UNITS = 128
NUM_BLOCKS = 2
NUM_HEADS = 2
DROPOUT = 0.2
DEVICE = torch.device('cuda')
MODEL_PATH = "sasrec_epoch_20.pth" 

class SASRec(nn.Module):
    def __init__(self, item_num, hidden_units, max_len, num_blocks, num_heads, dropout):
        super(SASRec, self).__init__()
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_units, nhead=num_heads, 
                dim_feedforward=hidden_units*4, dropout=dropout,
                batch_first=True, norm_first=True
            ) for _ in range(num_blocks)
        ])
        self.last_norm = nn.LayerNorm(hidden_units)

    def forward(self, seqs):
        seqs_emb = self.item_emb(seqs)
        positions = torch.arange(seqs.shape[1], device=seqs.device).unsqueeze(0)
        seqs_emb += self.pos_emb(positions)
        seqs_emb = self.emb_dropout(seqs_emb)
        L = seqs.shape[1]
        mask = torch.triu(torch.ones(L, L, device=seqs.device) * float('-inf'), diagonal=1)
        for block in self.blocks:
            seqs_emb = block(seqs_emb, src_mask=mask)
        seqs_emb = self.last_norm(seqs_emb)
        return seqs_emb

def predict(model, internal_id_list, item_num, k=5):
    model.eval()
    with torch.no_grad():
        seq = np.zeros([1, MAX_LEN], dtype=np.int32)
        curr_seq = internal_id_list[-MAX_LEN:]
        seq[0, -len(curr_seq):] = curr_seq
        
        seq_tensor = torch.LongTensor(seq).to(DEVICE)
        
        seq_emb = model(seq_tensor)
        last_emb = seq_emb[:, -1, :] 
        logits = last_emb @ model.item_emb.weight.t()
        
        logits[0, curr_seq] = -float('inf')
        logits[0, 0] = -float('inf')
        
        scores, indices = torch.topk(logits, k)
        return indices.cpu().numpy()[0]

def main():
    print("1. Loading dataset mappings")
    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    item2id = data['item2id']
    item_num = data['item_num']
    
    sample_key = next(iter(item2id))
    id_is_string = isinstance(sample_key, str)
    
    print(f"   Detected Dataset ID Type: {'STRING' if id_is_string else 'INTEGER'}")

    internal_to_raw = {v: k for k, v in item2id.items()}

    print("2. Loading Model")
    model = SASRec(item_num, HIDDEN_UNITS, MAX_LEN, NUM_BLOCKS, NUM_HEADS, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print("\n" + "="*40)
    print("INFERENCE MODE")
    print("="*40)
    print("Enter a list of Goodreads Book IDs")
    print("Type 'q' to quit.")
    
    while True:
        user_input = input("\nIDs > ").strip()
        
        if user_input.lower() == 'q':
            break
            
        try:
            # 1. Parse into a list of strings first
            raw_parts = [x.strip() for x in user_input.split(',')]
            
            internal_ids = []
            for part in raw_parts:
                # 2. Convert based on detected type
                if id_is_string:
                    lookup_key = str(part)
                else:
                    lookup_key = int(part)
                
                # 3. Lookup
                if lookup_key in item2id:
                    internal_ids.append(item2id[lookup_key])
                else:
                    print(f"Ignored unknown ID: {lookup_key}")
            
            if not internal_ids:
                print("No valid IDs provided.")
                continue
                
            # Predict
            recs = predict(model, internal_ids, item_num, k=5)
            
            print("\nRecommended Book IDs:")
            for rank, internal_idx in enumerate(recs):
                raw_id = internal_to_raw.get(internal_idx, "Unknown")
                print(f" {rank+1}. {raw_id}")
                
        except ValueError:
            print("Invalid format.")

if __name__ == '__main__':
    main()