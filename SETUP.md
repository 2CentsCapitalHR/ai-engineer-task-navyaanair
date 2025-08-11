# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/2CentsCapitalHR/ai-engineer-task-navyaanair
```

## 2. Install Requirements

Make sure you have **Python 3.10+** installed, then install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Model Access

This project uses the Gemma 3 (1B) model.
You will need an Ollama installation to run it locally.

* Install Ollama:
  * macOS / Linux: https://ollama.com/download
  * Windows: Download from the same link and follow setup instructions
* Pull the model:

```bash
ollama pull gemma3:1b
```

## 4. Prebuilt Embeddings

The repository already includes the ChromaDB folder with all document embeddings.
No additional embedding step is required.

If you want to rebuild embeddings from scratch:

```bash
python build_embeddings.py
```

## 5. Run the App

Start the Corporate Agent with:

```bash
python corporate_agent.py
```

This will open a Gradio interface in your browser.

## 6. Using the App

* Upload Document and get resulting JSON file
