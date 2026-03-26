import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
INDEX_NAME = "legal-rag-index-filtered"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    # 1. Read Pinecone API key from environment for security.
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        print("Error: Set PINECONE_API_KEY in your environment before running this script.")
        return

    # 2. Get the search query from the user
    print("\n" + "="*50)
    print("      LEGAL RAG SEARCH SYSTEM")
    print("="*50)
    query_text = input("Enter your legal question: ")
    
    if not query_text.strip():
        print("Empty query. Exiting.")
        return

    # 3. Load the embedding model (to convert your text question into numbers)
    print("\nLoading AI model to understand the question...")
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(query_text).tolist()

    # 4. Connect to Pinecone
    print("Searching the database for the most relevant laws...\n")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    # 5. Perform the vector search
    # include_metadata=True is crucial so we get our text back, not just the IDs!
    search_results = index.query(
        vector=query_embedding,
        top_k=3, # How many text chunks to return
        include_metadata=True 
    )

    # 6. Display the results
    print("="*50)
    print(f"TOP 3 RESULTS FOR: '{query_text}'")
    print("="*50 + "\n")
    
    for i, match in enumerate(search_results['matches'], start=1):
        score = match['score']
        metadata = match['metadata']
        
        # Format the score as a percentage
        match_percentage = round(score * 100, 2)
        
        print(f"RESULT {i} - Relevance: {match_percentage}%")
        print(f"Document: {metadata.get('title', 'Unknown')}")
        print(f"File:     {metadata.get('source_filename', 'Unknown')}")
        print("-" * 50)
        
        # Print a clean snippet of the text
        text_snippet = metadata.get('chunk_text', 'No text found')
        print(text_snippet)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
