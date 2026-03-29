# How to Run the Sindh Criminal Law RAG App (Windows)

## Prerequisites

- Python 3.10+ installed
- Git installed

## Setup (one-time)

1. **Clone the repo:**
   ```
   git clone https://github.com/rameezhoda21/NLP-mini-project.git
   cd NLP-mini-project
   ```

2. **Create a virtual environment:**
   ```
   python -m venv myenv
   myenv\Scripts\activate
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Create your `.env` file:**
   - Copy `.env.example` to `.env`
   - Fill in your API keys:
     ```
     PINECONE_API_KEY=your_pinecone_key
     GROQ_API_KEY=your_groq_key
     HF_API_TOKEN=your_hf_token
     ```
   - Get a free Groq key at https://console.groq.com

## Running the App

1. **Activate the environment:**
   ```
   myenv\Scripts\activate
   ```

2. **Launch:**
   ```
   streamlit run app.py
   ```

3. **Open** http://localhost:8501 in your browser.

## Using the App

- Type a legal question in the text box (e.g., *"What is the definition of Prosecutor under the Sindh Criminal Prosecution Service Act 2009?"*)
- Choose a retrieval mode from the sidebar: **Hybrid** (recommended), Semantic, or BM25
- Toggle **"Run live evaluation"** in the sidebar to see Faithfulness and Relevancy scores
- Click **Submit** and wait for the answer
- Expand the **Retrieved Context** sections to see which legal chunks were used
