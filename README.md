# Agentic Document Intake & Auto-Fill System

This project uses an **LLM-powered intake assistant** to automatically extract information from the user, validate inputs, and fill PDF forms such as USCIS, employment, or custom forms.

---

## 🚀 Features
- Intelligent Q&A to collect form data.
- Automatic PDF field mapping using PyMuPDF.
- Supports both AcroForm and non-fillable PDFs.
- Natural conversation handling (yes/no/N/A logic).
- Local execution (no web upload needed).

---

## 🧠 Requirements
- **Python 3.9 +**
- OpenAI Python SDK  
- PyMuPDF (`fitz`)  
- dateutil  
- httpx  

Install everything:
```bash
pip install openai PyMuPDF python-dateutil httpx

## How to Run

- Place your input form in the project folder — e.g. form.pdf
- Run the script from the command line:
python agentic_doc_intake_fill_align_again.py "C:\Users\paul\chatgpt-api-demo\form.pdf" "C:\Users\paul\chatgpt-api-demo\filled.pdf"
form.pdf → the blank document form you want to fill
filled.pdf → the automatically completed output file

## Project Structure
chatgpt-api-demo/
│
├── agentic_doc_intake_fill_align_again.py   # Main LLM + PDF logic
├── form.pdf                                 # Input (fillable or plain) form
├── filled.pdf                               # Output auto-filled document
└── README.md                                # Documentation



