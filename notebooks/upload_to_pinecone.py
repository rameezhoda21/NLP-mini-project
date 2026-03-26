import os
import time
import pandas as pd
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
INPUT_FILE = Path("data/rag_chunks_with_embeddings.pkl")
FLAG_FILE = Path("data/rag_chunks_with_retrievable_flag.csv")
INDEX_NAME = "legal-rag-index-filtered" # Use a separate index to keep filtered vectors isolated
BATCH_SIZE = 100               # Recommended batch size for Pinecone uploads
USE_FILTERED_CHUNKS = True      # If True, keep only rows where is_retrievable == True

def main():
    # 1. Insert your Pinecone API key here, inside the quotes
    api_key = "pcsk_5CRZGj_LTeaDvHWi3YbW6uWFXn2hfuTdRaXwWQdNp2knJ4LSTqfYnued3R3BBcPRxhErS1"
    
    if api_key == "PUT_YOUR_API_KEY_HERE":
        print("Error: You need to replace 'PUT_YOUR_API_KEY_HERE' with your actual Pinecone API key in the script.")
        return

    # 2. Load the pickle file
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_pickle(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}.")
        return

    if df.empty:
        print("Error: The dataframe is empty.")
        return

    # 2b. Optionally filter embeddings using the is_retrievable flag file.
    if USE_FILTERED_CHUNKS:
        if not FLAG_FILE.exists():
            print(f"Error: Could not find filter file {FLAG_FILE}.")
            return

        flags_df = pd.read_csv(FLAG_FILE)
        required_cols = {"chunk_id", "is_retrievable"}
        if not required_cols.issubset(flags_df.columns):
            print(
                "Error: Filter file must contain columns "
                f"{sorted(required_cols)}. Found: {list(flags_df.columns)}"
            )
            return

        retrievable_ids = set(
            flags_df.loc[flags_df["is_retrievable"] == True, "chunk_id"].astype(str)
        )

        before_count = len(df)
        df["chunk_id"] = df["chunk_id"].astype(str)
        df = df[df["chunk_id"].isin(retrievable_ids)].copy()
        after_count = len(df)

        print(
            f"Filtering enabled: kept {after_count} of {before_count} vectors "
            f"using {FLAG_FILE}."
        )

        if df.empty:
            print("Error: No rows left after filtering. Nothing to upload.")
            return

    total_chunks = len(df)
    # Dynamically find the dimension of the embedding (384 for all-MiniLM-L6-v2)
    embedding_dim = len(df.iloc[0]['embedding'])
    print(f"Loaded {total_chunks} chunks. Detected embedding dimension: {embedding_dim}")

    # 3. Initialize Pinecone client
    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=api_key)

    # 4. Create an index if it does not exist
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Index '{INDEX_NAME}' not found. Creating it now...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=embedding_dim,
            metric="cosine", # Cosine similarity is standard for sentence-transformers
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Change this if your Pinecone project is in a different region
            )
        )
        # Wait a moment for the index to be fully initialized
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Index created successfully!")
    else:
        print(f"Index '{INDEX_NAME}' already exists. Preparing to upload to it.")

    # Connect to the specific index
    index = pc.Index(INDEX_NAME)

    # 5. Format the data for Pinecone
    print("\nPreparing vectors for upload...")
    vectors = []
    
    for _, row in df.iterrows():
        # Clean metadata (Pinecone does not accept NaN/Null values in metadata)
        title = str(row.get('title', 'Unknown Title'))
        doc_id = str(row.get('doc_id', 'Unknown Doc ID'))
        source_filename = str(row.get('source_filename', 'Unknown File'))
        
        # Build the vector payload
        vector = {
            "id": str(row['chunk_id']),
            "values": row['embedding'],
            "metadata": {
                "chunk_text": row['chunk_text'],
                "title": title,
                "doc_id": doc_id,
                "source_filename": source_filename,
                "is_retrievable": True
            }
        }
        vectors.append(vector)

    # 6 & 7. Batch uploads for efficiency
    print(f"Starting upload in batches of {BATCH_SIZE}...")
    
    # Iterate over the vectors in chunks of BATCH_SIZE
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        
        # Upsert the batch to Pinecone
        index.upsert(vectors=batch)
        
        print(f"  • Uploaded [{min(i + BATCH_SIZE, total_chunks)} / {total_chunks}] vectors")

    # 8. Print progress and final count
    # Let pinecone update its stats before fetching them
    time.sleep(2) 
    stats = index.describe_index_stats()
    
    print("\n" + "="*35)
    print("           UPLOAD COMPLETE")
    print("="*35)
    print(f"Total rows in DataFrame:          {total_chunks}")
    print(f"Total vectors in Pinecone Index:  {stats.total_vector_count}")
    print("="*35)

if __name__ == "__main__":
    main()
