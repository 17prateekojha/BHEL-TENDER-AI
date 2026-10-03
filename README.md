# Document RAG Assistant

Answers questions **only from the files in `docs/`**, formats them cleanly, explains *why*, refuses unethical queries, and offers a BHEL-branded browser portal alongside the VS Code extension.

## Browser portal

Run the Streamlit portal locally after installing the dependencies:

```bash
streamlit run app.py
```

The portal uses the BHEL logo embedded in the tender document, supports document re-indexing, and displays source file and page/line citations. Configure either `ANTHROPIC_API_KEY` or `GROQ_API_KEY` as an environment variable for local use.

### Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push this project, including `docs/` and `assets/bhel-logo.png`. Do not upload `.venv/`, `.rag_index/`, `.env`, or `.streamlit/secrets.toml`; these are excluded by `.gitignore`.
2. Confirm that the tender files are approved for public access and third-party AI processing. A public portal lets anyone ask questions about the included documents, and matching text is sent to the configured model provider.
3. Sign in at [share.streamlit.io](https://share.streamlit.io), choose **Create app**, select the GitHub repository and branch, set the main file to `app.py`, then deploy. The app URL will be public to anyone who has it.
4. Open the app's **Settings > Secrets** and set Groq:

	```toml
	RAG_PROVIDER = "groq"
	GROQ_API_KEY = "your-valid-key"
	```

	Never commit API keys or `.streamlit/secrets.toml`.

The online instance builds its vector index from `docs/`; its local index storage is temporary and can be rebuilt. The existing FastAPI API and VS Code extension remain separate from this browser portal.

Scanned PDFs are OCR'd in English during indexing. Streamlit Community Cloud installs Tesseract from `packages.txt`; local Windows runs also need Tesseract OCR installed and available on `PATH`.

## Project layout

```
rag-assistant/
├── PRD.md                  requirements
├── rag.py                  engine: ingest, retrieve, guardrail, answer
├── server.py               local API for VS Code
├── requirements.txt
├── docs/                   put your documents here (pdf, docx, md, txt, code)
└── vscode-extension/       the VS Code extension (TypeScript)
```

## 1. Run the backend

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"   # Windows: set ANTHROPIC_API_KEY=your-key
# To use Groq instead, set RAG_PROVIDER=groq and GROQ_API_KEY.

python rag.py --reindex               # optional: build the index now
python rag.py "How does login work?"  # test from the terminal
uvicorn server:app --port 8000        # start the API for VS Code
```

Settings (environment variables): `RAG_DOCS_DIR` (default `./docs`), `RAG_LLM_MODEL`, `RAG_TOP_K`, `RAG_MIN_SCORE`.
New, changed or deleted files are detected automatically on the next question.

## 2. Install the VS Code extension

```bash
cd vscode-extension
npm install
npm run compile
```

Then open the `vscode-extension` folder in VS Code and press **F5**. A new window opens with the extension loaded. To install it permanently, run `npx @vscode/vsce package` and install the generated `.vsix`.

Commands (Ctrl+Shift+P):

| Command | What it does |
|---|---|
| **RAG: Ask Documents** (Ctrl+Alt+R) | Opens a formatted answer with justification and clickable sources |
| **RAG: Fetch Code Snippet** | Lists matching code from your documents and inserts your pick at the cursor |
| **RAG: Re-index Documents** | Forces a rebuild of the index |

## 3. The PRD in simple terms

**The problem:** your documents hold the answers, but searching them by hand is slow, and normal AI chatbots may make things up or can't show where an answer came from.

**What we are building:** an assistant that reads your folder and answers questions from it. Think of it as an open-book exam where the only book allowed is your folder.

| Requirement | What it means in practice | Where it lives in the code |
|---|---|---|
| 1. Answers from the folder only | The AI is shown only the matching passages. If nothing matches, it says the documents don't contain the answer instead of guessing. | `retrieve()`, `ANSWER_SYSTEM` |
| 2. Clean, proper format | Every answer has the same layout: Answer, Details, Justification, Sources. | `answer()` |
| 3. Unethical questions | A safety check runs first. If the question is harmful, the reply is exactly `I AM UNABLE TO ANSWER YOUR QUERY`, nothing more. | `is_unethical()` |
| 4. Clear justification | The AI explains how the passages support the answer and lists the file and page/line numbers. | `justification` field, Sources list |
| 5. Code in VS Code | Code from your documents is returned in code blocks, and you can insert it at your cursor or click a source to open the file at that line. | `snippets`, `extension.ts` |

**Key terms**

- **RAG:** the assistant first *retrieves* relevant passages, then *generates* an answer from them.
- **Embeddings / vector index:** a way of turning text into numbers so that passages with similar meaning can be found quickly.
- **Guardrail:** a safety check on the question before anything else happens.
- **Grounded:** every claim in the answer can be traced to a passage in your documents.

## Limitations to be aware of

- The ethics check is an AI classifier, so it is not perfect. Test it with your own list of prompts and adjust `GUARD_SYSTEM` in `rag.py`. If the check itself fails (for example a missing API key), the app refuses by default.
- Scanned PDFs are OCR'd in English; Tesseract OCR must be installed and available on `PATH` for local indexing.
- Document content is sent to the Anthropic API when answering. For fully offline use, swap in a local model.
