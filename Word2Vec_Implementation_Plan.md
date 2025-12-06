# Word2Vec Model Implementation Plan for Book Recommendation System

A step-by-step guide to implementing a Word2Vec model from scratch.

---

## Phase 1: Data Preparation

### Step 1: Load Preprocessed Data
- Load user reading sequences (books read by each user in chronological order)
- Each sequence represents a user's reading history as a list of book IDs

### Step 2: Build Vocabulary
- Create a mapping of book IDs to indices (`book_to_idx`)
- Create reverse mapping of indices to book IDs (`idx_to_book`)
- Calculate vocabulary size for embedding matrix dimensions

### Step 3: Generate Training Pairs
- Use a sliding window approach to create (context, target) pairs
- For each book in a sequence, pair it with surrounding books within the window
- Example with window size 2: `[A, B, C, D, E]` → pairs for C: `(C,A), (C,B), (C,D), (C,E)`

---

## Phase 2: Model Architecture

### Step 4: Initialize Embeddings
Create two weight matrices:
- **Input embeddings (W1)**: Shape `(vocab_size, embedding_dim)` - book → hidden layer
- **Output embeddings (W2)**: Shape `(embedding_dim, vocab_size)` - hidden layer → book
- Initialize with small random values (e.g., uniform distribution [-0.5/dim, 0.5/dim])

### Step 5: Choose Architecture

#### Option A: Skip-gram (Recommended)
- **Input**: Target book
- **Output**: Predict context books
- Better for rare books, captures more nuanced relationships

#### Option B: CBOW (Continuous Bag of Words)
- **Input**: Context books (averaged)
- **Output**: Predict target book
- Faster to train, good for frequent books

---

## Phase 3: Training Components

### Step 6: Implement Forward Pass
```
1. Look up input embedding for target book
2. Compute dot product with output embeddings
3. Apply activation function (softmax or sigmoid)
4. Output probability distribution over vocabulary
```

### Step 7: Implement Loss Function

#### Option A: Full Softmax (Not recommended for large vocabulary)
- Compute probability over entire vocabulary
- Very expensive for large book collections

#### Option B: Negative Sampling (Recommended)
- For each positive pair, sample K negative examples
- Binary classification: distinguish real pairs from fake pairs
- Loss: `-log(σ(v_pos · u)) - Σ log(σ(-v_neg · u))`

#### Option C: Hierarchical Softmax
- Organize vocabulary in binary tree
- Reduces complexity from O(V) to O(log V)

### Step 8: Implement Backpropagation
- Compute gradients of loss with respect to embeddings
- Update both input and output embeddings
- Gradient for positive sample: `(σ(score) - 1) * embedding`
- Gradient for negative sample: `σ(score) * embedding`

### Step 9: Add Optimization
- **SGD**: Simple, effective with learning rate decay
- **Adam**: Adaptive learning rates, faster convergence
- Implement learning rate scheduling (linear decay recommended)

---

## Phase 4: Training Loop

### Step 10: Batch Processing
- Shuffle training pairs each epoch
- Create mini-batches for efficient computation
- Typical batch size: 128-512

### Step 11: Training Loop
```python
for epoch in range(num_epochs):
    for batch in batches:
        # Forward pass
        # Compute loss
        # Backward pass
        # Update weights
    # Log progress & metrics
```

---

## Phase 5: Evaluation & Usage

### Step 12: Extract Book Vectors
- Use input embedding matrix (W1) as final book vectors
- Each row represents a book's learned embedding
- Save embeddings for later use in recommendations

### Step 13: Implement Similarity Search
- Use cosine similarity to find similar books:
  ```
  similarity(a, b) = (a · b) / (||a|| * ||b||)
  ```
- For a given book, return top-K most similar books

### Step 14: Evaluate Quality
- **Qualitative**: Check if similar books make sense (same genre, author, theme)
- **Quantitative**: 
  - Hit rate on held-out user sequences
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (nDCG)

---

## Recommended Configuration

### Architecture
- **Model**: Skip-gram with Negative Sampling
- **Reason**: Works well with sparse data and captures book relationships effectively

### Hyperparameters

| Parameter | Recommended Value | Range to Explore |
|-----------|------------------|------------------|
| Embedding dimension | 128 | 64-300 |
| Window size | 5 | 3-10 |
| Negative samples | 10 | 5-20 |
| Learning rate | 0.025 | 0.01-0.05 |
| Min learning rate | 0.0001 | - |
| Epochs | 10 | 5-20 |
| Batch size | 256 | 128-512 |
| Min count (vocabulary) | 5 | 1-10 |

---

## Implementation Checklist

- [ ] **Phase 1: Data Preparation**
  - [ ] Load user sequences
  - [ ] Build vocabulary mappings
  - [ ] Generate training pairs with sliding window

- [ ] **Phase 2: Model Architecture**
  - [ ] Initialize input embeddings (W1)
  - [ ] Initialize output embeddings (W2)
  - [ ] Implement Skip-gram architecture

- [ ] **Phase 3: Training Components**
  - [ ] Implement forward pass
  - [ ] Implement negative sampling
  - [ ] Implement gradient computation
  - [ ] Implement weight updates

- [ ] **Phase 4: Training Loop**
  - [ ] Create batch generator
  - [ ] Implement main training loop
  - [ ] Add progress logging
  - [ ] Implement learning rate decay

- [ ] **Phase 5: Evaluation & Usage**
  - [ ] Extract and save book embeddings
  - [ ] Implement similarity search function
  - [ ] Test with sample queries
  - [ ] Evaluate recommendation quality

---

## File Structure

```
Book-Recommendation-Model/
├── Word2Vec_Scratch.py          # Main implementation file
├── Word2Vec_Implementation_Plan.md  # This plan
├── Dataset_half/
│   └── processed_artifacts/
│       ├── user_sequences/      # Input: user reading sequences
│       └── book_metadata_lookup/ # Book metadata for evaluation
└── Word2Vec_scratch_model/      # Output: trained model & embeddings
    ├── embeddings.npy
    ├── vocabulary.json
    └── model_config.json
```

---

## Next Steps

1. Review the preprocessed data format in `Dataset_half/processed_artifacts/`
2. Start implementing Phase 1 in a new `Word2Vec_Scratch.py` file
3. Test each component before moving to the next phase
4. Iterate on hyperparameters based on evaluation results
