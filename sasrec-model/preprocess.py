import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import pickle
import os

REVIEWS_FILE = './goodreads-book-reviews/goodreads_reviews_dedup.json'
BOOKS_METADATA = './goodreads-book-reviews/goodreads_books.json'

# Filter the data, user must read at least 5 books
MIN_HISTORY_LENGTH = 5

# Timestamps are essential for SASRec to build the sequence history
def parse_time(date_str):
    if not date_str: return None
    try:
        parts = date_str.split()
        # ignores the timezone
        new_str = f"{parts[1]} {parts[2]} {parts[3]} {parts[5]}"
        return datetime.strptime(new_str, "%b %d %H:%M:%S %Y").timestamp()
    except:
        return None
    
def preprocess():
    data = []

    with open(REVIEWS_FILE, 'r') as f:
        for line in tqdm(f, desc="Parsing JSON"):
            record = json.loads(line)

            time_str = record.get('read_at', '')
            if not time_str:
                time_str = record.get('date_added', '')

            ts = parse_time(time_str)
            if ts:
                data.append([record['user_id'], record['book_id'], ts])

    # Convert to DataFrame for filtering
    df = pd.DataFrame(data, columns=['user_id', 'book_id', 'timestamp'])
    print(f"Total raw interactions: {len(df)}")

    df = df.drop_duplicates(subset=['user_id', 'book_id'])

    # Filter users with short reading history
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= MIN_HISTORY_LENGTH].index
    df = df[df['user_id'].isin(valid_users)]

    print(f"Filtered interactions (Minimum w/ {MIN_HISTORY_LENGTH} books): {len(df)}")
    print(f"Unique users: {df['user_id'].unique()}")
    print(f"Unique books: {df['book_id'].unique()}")

    # Mapping
    user2id = {u: i+1 for i, u in enumerate(df['user_id'].unique())}
    item2id = {i: x+1 for x, i in enumerate(df['book_id'].unique())}

    df['user_idx'] = df['user_id'].map(user2id)
    df['item_idx'] = df['book_id'].map(item2id)

    # Building sequence
    df = df.sort_values(['user_idx', 'timestamp'])
    user_train = df.groupby('user_idx')['item_idx'].apply(list).to_dict()

    # Export dataset
    with open('dataset.pkl', 'wb') as f:
        pickle.dump({
            'user_train': user_train,
            'user2id': user2id,
            'item2id': item2id,
            'item_num': len(item2id),
            'user_num': len(user2id)
        }, f)

if __name__ == '__main__':
    preprocess()