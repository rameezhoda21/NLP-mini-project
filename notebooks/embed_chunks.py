import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Configuration & File Paths
# ---------------------------------------------------------
# Define paths using pathlib for cross-platform compatibility
BASE_DIR = Path("data")
INPUT_CSV = BASE_DIR / "rag_chunks.csv"
OUTPUT_FILE = BASE_DIR / "rag_chunks_with_embeddings.pkl"

# Define the model to use for embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    # 1. Read the chunk CSV file
    print(f"Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}. Please check the path.")
        return

    initial_count = len(df)

    # 2. Clean the data (Track and drop empty chunks to avoid model errors)
    # Convert purely whitespace fields to NaN, then drop NaNs in 'chunk_text'
    df['chunk_text'] = df['chunk_text'].replace(r'^\s*$', pd.NA, regex=True)
    df_clean = df.dropna(subset=['chunk_text']).copy()
    
    skipped_count = initial_count - len(df_clean)
    
    if len(df_clean) == 0:
        print("Error: No valid chunks found to embed. Exiting.")
        return

    print(f"Found {len(df_clean)} valid chunks. (Skipped {skipped_count} empty chunks)")

    # 3. Load the embedding model
    print(f"\nLoading embedding model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. Generate embeddings
    # Extract the text column to a standard python list for the model
    texts = df_clean['chunk_text'].tolist()
    
    print("Generating embeddings... (This might take a few moments)")
    # .encode() processes the texts efficiently and returns a 2D numpy array
    embeddings_array = model.encode(texts, show_progress_bar=True)
    
    # 5. Store the embeddings along with the original metadata
    # We convert the 2D numpy array into a list of 1D arrays so it fits perfectly in a single pandas column
    df_clean['embedding'] = embeddings_array.tolist()
    
    # 6. Save the result to a local file
    print(f"\nSaving embedded chunks to {OUTPUT_FILE}...")
    # Using pickle (.pkl) format preserves the list structure of the embeddings natively
    # You can easily read this later using: pd.read_pickle("data/rag_chunks_with_embeddings.pkl")
    df_clean.to_pickle(OUTPUT_FILE)
    
    # 7. Print the final summary
    embedding_dim = len(df_clean.iloc[0]['embedding'])
    print("\n" + "="*30)
    print("           SUMMARY")
    print("="*30)
    print(f"Total chunks processed: {len(df_clean)}")
    print(f"Rows skipped (empty):   {skipped_count}")
    print(f"Embedding dimension:    {embedding_dim}")
    print(f"Output saved to:        {OUTPUT_FILE}")
    print("Status:                 Ready for Pinecone upload!")
    print("="*30)

if __name__ == "__main__":
    main()