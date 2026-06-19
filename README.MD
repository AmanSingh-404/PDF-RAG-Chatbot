# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that lets you upload a PDF document and ask questions about its content. The app retrieves the most relevant chunks from the document and uses a large language model to answer strictly based on that retrieved context, reducing hallucinated answers.

The project includes both a command-line version and a Streamlit-based web interface.

## How It Works

The application follows a standard RAG pipeline:

1. **Document Loading**: A PDF is loaded and its text content is extracted page by page.
2. **Chunking**: The extracted text is split into smaller overlapping chunks so that the embedding model can process meaningful, retrievable units of text instead of entire pages.
3. **Embedding**: Each chunk is converted into a numerical vector using a sentence-transformer embedding model.
4. **Vector Storage**: These vectors are stored in a Chroma vector database, which allows fast similarity-based search.
5. **Retrieval**: When a user asks a question, the question is embedded and compared against the stored vectors using Maximal Marginal Relevance (MMR) search, which returns chunks that are both relevant and diverse, reducing redundant context.
6. **Answer Generation**: The retrieved chunks are passed as context to a Mistral AI chat model along with the original question. The model is instructed to answer only from the provided context, and to explicitly say so if the answer is not present.

## Project Structure

```
.
├── app.py                  # Streamlit web UI (upload PDF, build index, chat)
├── main.py                 # Command-line RAG chatbot (reads from an existing chromaDB)
├── createdatabase.py       # Script to build a Chroma vector store from a local PDF
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

### app.py

The main entry point for most users. This is a Streamlit application that:

- Provides a sidebar file uploader for PDF documents.
- Processes the uploaded PDF (loads, splits, embeds, and stores it in Chroma) when the user clicks "Process document".
- Rebuilds the Chroma collection fresh on every new upload, so that content from a previous document does not leak into the current session.
- Presents a chat interface where the user can ask questions about the uploaded document.
- Displays the retrieved context chunks (with page numbers) in an expandable section under each answer, so you can verify what the model based its answer on.

### main.py

A simpler, terminal-based version of the same chatbot. It assumes a Chroma vector store already exists on disk (created via `createdatabase.py`) and lets you chat with it directly from the command line. Useful for quick testing without running the Streamlit UI.

### createdatabase.py

A standalone script that builds the Chroma vector store from a single hardcoded PDF path. This is what `main.py` depends on, and is essentially the same logic that `app.py` runs internally whenever a new file is uploaded through the web interface.

## Prerequisites

- Python 3.10 or higher
- A Mistral AI API key (used for the chat model)

## Setup

1. Clone the repository:

   ```
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment (recommended):

   ```
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Mistral API key:

   ```
   MISTRAL_API_KEY=your_api_key_here
   ```

## Usage

### Option 1: Streamlit Web UI (recommended)

```
streamlit run app.py
```

This will open the app in your browser. From there:

1. Upload a PDF using the sidebar.
2. Click "Process document" and wait for indexing to complete.
3. Use the chat box at the bottom of the page to ask questions about the document.
4. Expand "View retrieved context" under any answer to see which chunks were used to generate it.

### Option 2: Command-line version

This requires a vector store to already exist. First, edit `createdatabase.py` to point to your PDF file path, then run it:

```
python createdatabase.py
```

Once the `chromaDB` folder has been created, run the chatbot:

```
python main.py
```

Type your questions at the prompt. Enter `0` to exit.

## Configuration Notes

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`, a lightweight model that runs locally without requiring an API call.
- **Chat model**: `mistral-small-2506`, accessed through the Mistral AI API.
- **Chunking**: 1000 characters per chunk with 200 characters of overlap, configurable in `createdatabase.py` or `app.py`.
- **Retrieval strategy**: MMR search with `k=4`, `fetch_k=10`, and `lambda_mult=0.5`, balancing relevance and diversity in the retrieved chunks.

These values can be tuned depending on the size and nature of the documents you work with.

## Limitations

- The vector store is rebuilt from scratch on every new upload in the Streamlit app, so previously processed documents are not retained between sessions.
- Answers are only as good as the retrieved context; very large or poorly structured PDFs may require tuning the chunk size or retrieval parameters.
- The app currently supports a single PDF at a time.


This project is provided as-is for personal and educational use.
