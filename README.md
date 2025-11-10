# Agentic Intake API (Intake-only)

Natural, multi-turn chatbot that reads a **single PDF**, interviews the user to collect all needed fields, and produces a **simple JSON**: `{ "field_name": "value", ... }`.
_No PDF filling here (intake only)._

---

## What you get

- `POST /intake/start` – upload a PDF, LLM extracts the fields (schema).
- `POST /intake/message` – chat to fill fields (handles “N/A”, validation).
- `POST /intake/set-values` – bulk set values programmatically.
- `GET /intake/json` – export collected values as `{field: value}`.
- `GET /intake/status` – progress (required-only and overall).
- `GET /health` – health check.
- Swagger: `/docs`, ReDoc: `/redoc`.

---

## Requirements

- Python **3.10+**
- An **OpenAI API key**
- Windows/Mac/Linux

---

## Quickstart

### 1) Create a virtual environment & install dependencies

**Windows (PowerShell):**
```powershell
cd C:\path\to\your\project
python -m venv venv
.env\Scripts\Activate.ps1
pip install fastapi uvicorn "pydantic<2" python-dotenv openai httpx PyMuPDF python-dateutil
```

**macOS/Linux:**
```bash
cd /path/to/your/project
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn "pydantic<2" python-dotenv openai httpx PyMuPDF python-dateutil
```

### 2) Add your OpenAI key (no hard-coding)

Create a file named **`.env`** in the project root:

```
OPENAI_API_KEY=sk-...your_openai_key...
```

> The app loads this via `python-dotenv`. Keep `.env` out of git.

### 3) Run the API

```bash
python -m uvicorn app:app --port 8000
```
- Swagger UI: <http://127.0.0.1:8000/docs>  
- ReDoc: <http://127.0.0.1:8000/redoc>  
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>

---

## Usage (cURL)

> Replace `C:\path\to\form.pdf` and `<SID>` with your values.

### 1) Start a session (upload PDF)
```bash
curl -F "file=@C:\path\to\form.pdf"      http://127.0.0.1:8000/intake/start
```
Response includes:
```json
{
  "session_id": "UUID...",
  "bot_message": "First questions...",
  "done": false,
  "progress": { "filled": 0, "total": 12, "percent": 0, "bar": "[------------------------]   0%", "basis": "required", "overall": {...} }
}
```

### 2) Chat values in (repeat until done)
```bash
curl -X POST http://127.0.0.1:8000/intake/message   -H "Content-Type: application/json"   -d "{"session_id":"<SID>","user_text":"First name Jane, last name Doe"}"
```

### 3) (Optional) Bulk set values
```bash
curl -X POST http://127.0.0.1:8000/intake/set-values   -H "Content-Type: application/json"   -d "{"session_id":"<SID>","values":{"email":"jane@doe.com","dob":"1990-05-01"}}"
```

### 4) Get final JSON
```bash
curl "http://127.0.0.1:8000/intake/json?session_id=<SID>&omit_nulls=false"
```

### 5) Check status/progress at any time
```bash
curl "http://127.0.0.1:8000/intake/status?session_id=<SID>"
```

---

## Postman / Insomnia

1. **Start**: `POST /intake/start`  
   - Body → **form-data**
     - key `file`, type **File**, choose your `form.pdf`
2. **Chat**: `POST /intake/message` – JSON body with `session_id` and `user_text`.
3. **Export**: `GET /intake/json?session_id=<SID>&omit_nulls=false`.

Import OpenAPI: paste `http://127.0.0.1:8000/openapi.json` into Postman.

---

## Behavior & Validation

- “N/A / not applicable / leave blank”:
  - **Boolean/checkbox** → coerced to `"No"`.
  - All other types → value omitted (`null` in JSON if `omit_nulls=false`).
- Dates parsed via `python-dateutil` with policy (e.g., DOB can’t be future).
- Emails/phones validated with simple regexes.

---

## Troubleshooting

- **Port in use (Windows WinError 10048)**  
  Find & kill process:
  ```bat
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  ```
  Or run on a different port: `--port 8001`.

- **`/docs` loading slowly**  
  Try `/redoc` or run without reload:
  ```bash
  python -m uvicorn app:app --port 8000
  ```

- **Empty/poor field extraction**  
  Ensure the PDF contains selectable text (not just scanned images).

- **OPENAI_KEY not picked up**  
  Confirm `.env` exists at project root and is spelled exactly `.env`.  
  Or set it in your shell:
  - **Windows PowerShell**: `$env:OPENAI_API_KEY="sk-..."`
  - **macOS/Linux**: `export OPENAI_API_KEY="sk-..."`

---

## .gitignore (recommended)

```
venv/
.env
__pycache__/
*.pyc
data/
*.pdf
*.log
```

---

## Example flow (concise)

1. `POST /intake/start` → get `session_id`  
2. `POST /intake/message` (repeat) or `POST /intake/set-values`  
3. `GET /intake/json`  
4. `GET /intake/status`
