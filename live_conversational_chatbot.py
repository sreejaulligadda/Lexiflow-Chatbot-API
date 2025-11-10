# app.py
# Intake-only API: Upload → chat → JSON (no PDF filling, no Cloudmersive)
# Run:
#   pip install fastapi uvicorn "pydantic<2" python-dotenv openai httpx PyMuPDF python-dateutil
#   python -m uvicorn app:app --reload --port 8000
#
# Requires OPENAI_API_KEY in environment (or .env with python-dotenv).

import os
import re
import json
import time
import uuid
from datetime import date
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from dateutil.parser import parse as parse_date
from dotenv import load_dotenv

# ===================== ENV/CONFIG =====================
load_dotenv()

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

client = OpenAI(timeout=60)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ===================== APP =====================
app = FastAPI(
    title="Agentic Intake API (Intake-only)",
    version="2.1.0",
    description="Upload PDF → chat/set-values → JSON (no document filling)",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "defaultModelExpandDepth": 0,
        "displayRequestDuration": True,
        "persistAuthorization": True
    },
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== UTILS =====================
NA_TOKENS = {
    "n/a","na","not applicable","none","no number","blank","skip",
    "leave blank","leave it blank","leave as blank"
}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^[\d\s()+-]{7,}$")

DATE_RULES = {
    "dob": {"future_ok": False, "allow_today": False},
    "birth": {"future_ok": False, "allow_today": False},
    "start": {"future_ok": False, "allow_today": True},
    "join": {"future_ok": False, "allow_today": True},
    "end": {"future_ok": True,  "allow_today": True},
    "expiry": {"future_ok": True, "allow_today": True},
    "signature": {"future_ok": False, "allow_today": True},
    "signed": {"future_ok": False, "allow_today": True},
}

def _norm(t: Any) -> str:
    s = t if isinstance(t, str) else str(t)
    return re.sub(r"\s+", " ", s.strip().lower())

def is_na(v: Any) -> bool:
    if v is None: return True
    if isinstance(v, str): return _norm(v) in NA_TOKENS
    return False

def call_openai(fn, max_retries: int = 5, backoff: int = 2):
    last = None
    for a in range(1, max_retries + 1):
        try:
            return fn()
        except (APIConnectionError, APITimeoutError,
                httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError,
                RateLimitError) as e:
            last = e
            time.sleep(backoff ** a)
    raise last

# ===================== PDF TEXT (schema discovery) =====================
def extract_plain(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        blocks = page.get_text("dict")["blocks"]
        spans = []
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    t = s.get("text", "")
                    if t.strip():
                        spans.append(t)
        pages.append(f"[PAGE {i+1}]\n" + " ".join(spans))
    doc.close()
    return "\n\n".join(pages)

# ===================== VALIDATION =====================
def pick_date_policy(field_name: str, field_label: str):
    key = _norm((field_name or "") + " " + (field_label or ""))
    for hint, rule in DATE_RULES.items():
        if hint in key: return rule
    return {"future_ok": False, "allow_today": True}

def norm_date_for_field(field_name: str, field_label: str, value: Any):
    try:
        dt = parse_date(str(value), dayfirst=True, yearfirst=False)
        today = date.today()
        pol = pick_date_policy(field_name, field_label)
        if dt.date() > today and not pol["future_ok"]:
            return None, "That date appears to be in the future."
        if dt.date() == today and not pol["allow_today"]:
            return None, "Please provide a date that is not today."
        return dt.strftime("%Y-%m-%d"), None
    except Exception:
        return None, "Use dd/mm/yyyy or mm/dd/yyyy."

def validate(field_meta: Dict[str, Any], val: Any):
    ftype = (field_meta.get("type", "text") or "text").lower()
    if is_na(val):
        if ftype in ("boolean","checkbox"): return "No", None   # N/A → No for booleans
        return None, None                                      # N/A → null for others
    if val is None or (isinstance(val, str) and not val.strip()):
        return None, "Missing value"
    name  = field_meta.get("name", "")
    label = field_meta.get("label", "")
    if ftype == "date":  return norm_date_for_field(name, label, val)
    if ftype == "email": return (val, None) if EMAIL_RE.match(str(val).strip()) else (None, "Invalid email")
    if ftype == "phone":
        v = re.sub(r"\s+"," ", str(val).strip())
        return (v, None) if PHONE_RE.match(v) else (None, "Invalid phone")
    if ftype in ("boolean","checkbox"):
        vv = _norm(val)
        if vv in ("yes","y","true","1","checked"):   return "Yes", None
        if vv in ("no","n","false","0","unchecked"): return "No", None
        return None, "Answer yes or no"
    return (str(val).strip() or None), None

# ===================== LLM PROMPTS =====================
SCHEMA_SYS = """You are an expert document-intake specialist.
From the provided document text, identify ALL fields a human would reasonably provide to complete the document.

Return STRICT JSON:
{
  "fields": [
    {
      "name": "snake_case_key",
      "label": "how printed in the doc",
      "type": "text|date|email|phone|boolean|checkbox|enum",
      "required": true|false,
      "anchor_phrase": "short phrase printed near the blank",
      "page": 1
    }
  ]
}
Rules:
- Include all likely user-supplied fields.
- Use 'boolean' for yes/no; 'checkbox' for visible checkboxes.
- If optional ('if applicable'), set required=false.
- Prefer concise names (snake_case).
"""

PLAN_SYS_TEMPLATE = """You are a warm, efficient intake agent.
Goal: naturally collect the remaining details for this document.
HARD RULES:
- Ask ONLY about FOCUS_FIELDS.
- Do NOT ask for fields already in KNOWN/LOCKED unless in NEEDS_CLARIFICATION.
- Treat values in KNOWN as final unless listed in NEEDS_CLARIFICATION.
- Respect N/A and move on.
Guidelines:
- Ask at most {max_q} short questions per turn.
- Group related items, avoid lists.
- Offer optional items after required.
Write only conversational text.
"""

EXTRACT_SYS = """You extract structured updates from the user's free-text reply.
Return STRICT JSON:
{
  "values": { "<field_name>": "<explicit_value_from_user>", ... },
  "confirm": ["<field_name>", ...],
  "reject": ["<field_name>", ...],
  "leave_blank": ["<field_name>", ...],
  "proposed": { "<field_name>": "<value_from_last_question>", ... }
}
"""

CONFIRM_SYS = """In 1–3 short lines, politely confirm the most recently added or corrected key values (plain text, no JSON)."""

def llm_schema(plain: str) -> List[Dict[str, Any]]:
    msgs=[{"role":"system","content":SCHEMA_SYS},
          {"role":"user","content":f"DOCUMENT_TEXT (first 10k chars):\n{plain[:10000]}"}]
    r = call_openai(lambda: client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, messages=msgs, response_format={"type":"json_object"}))
    try: return json.loads(r.choices[0].message.content).get("fields", [])
    except: return []

def llm_plan(known, invalid, focus_subset, max_q, have_required_done, offered_optional, locked=None):
    if locked is None: locked = []
    plan_sys = PLAN_SYS_TEMPLATE.format(max_q=max_q)
    intent = {"have_required_done": have_required_done, "have_offered_optional": offered_optional}
    msgs=[{"role":"system","content":plan_sys},
          {"role":"user","content":(
              "KNOWN:\n"+json.dumps(known,ensure_ascii=False)+
              "\n\nLOCKED:\n"+json.dumps(list(locked),ensure_ascii=False)+
              "\n\nNEEDS_CLARIFICATION:\n"+json.dumps(invalid,ensure_ascii=False)+
              "\n\nFOCUS_FIELDS:\n"+json.dumps(focus_subset,ensure_ascii=False)+
              "\n\nINTENT:\n"+json.dumps(intent)
          )}]
    r = call_openai(lambda: client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, messages=msgs))
    return r.choices[0].message.content.strip()

def llm_extract(schema, known, last_q, user_reply):
    if not user_reply or not str(user_reply).strip():
        return {"values":{}, "confirm":[], "reject":[], "leave_blank":[], "proposed":{}}
    msgs=[{"role":"system","content":EXTRACT_SYS},
          {"role":"user","content":f"SCHEMA:{json.dumps(schema,ensure_ascii=False)}\nKNOWN:{json.dumps(known,ensure_ascii=False)}\nLAST_QUESTION:{last_q}\nREPLY:{user_reply}"}]
    r = call_openai(lambda: client.chat.completions.create(
        model=MODEL, temperature=0, messages=msgs, response_format={"type":"json_object"}))
    try: raw = json.loads(r.choices[0].message.content)
    except: raw = {}
    return {
        "values": raw.get("values",{}) or {},
        "confirm": raw.get("confirm",[]) or [],
        "reject": raw.get("reject",[]) or [],
        "leave_blank": raw.get("leave_blank",[]) or [],
        "proposed": raw.get("proposed",{}) or {}
    }

def llm_confirm(new_info):
    if not new_info: return ""
    msgs=[{"role":"system","content":CONFIRM_SYS},
          {"role":"user","content":json.dumps(new_info,ensure_ascii=False)}]
    r = call_openai(lambda: client.chat.completions.create(model=MODEL, temperature=0.2, messages=msgs))
    return r.choices[0].message.content.strip()

# ===================== JSON + PROGRESS =====================
def build_simple_map(fields: List[Dict[str, Any]], known: Dict[str, Any], omit_nulls: bool=False) -> Dict[str, Any]:
    out = {}
    for f in fields:
        name = f["name"]
        val = known.get(name, None)
        val = (None if val in (None, "") else val)
        if omit_nulls and val is None:
            continue
        out[name] = val
    return out

def write_simple_json(json_path: str, fields: List[Dict[str, Any]], known: Dict[str, Any], omit_nulls: bool):
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    data = build_simple_map(fields, known, omit_nulls=omit_nulls)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data

def _make_bar(pct: int, width: int = 24) -> str:
    filled_chars = int(round((pct/100) * width))
    return "[" + "#" * filled_chars + "-" * (width - filled_chars) + f"] {pct:>3d}%"

def progress(fields: List[Dict[str, Any]], known: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    total_req = len(required) if required else 0
    filled_req = sum(1 for n in required if (n in known and known[n] not in (None, ""))) if total_req else 0
    pct_req = 0 if total_req == 0 else int(round(100 * filled_req / total_req))
    total_all = len(fields)
    filled_all = sum(1 for f in fields if (f["name"] in known and known[f["name"]] not in (None, "")))
    pct_all = 0 if total_all == 0 else int(round(100 * filled_all / total_all))
    return {
        "filled": filled_req,
        "total": total_req,
        "percent": pct_req,
        "bar": _make_bar(pct_req),
        "basis": "required",
        "overall": {
            "filled": filled_all,
            "total": total_all,
            "percent": pct_all,
            "bar": _make_bar(pct_all),
            "basis": "all_fields"
        }
    }

# ===================== MODELS =====================
class MessageIn(BaseModel):
    session_id: str
    user_text: str

class SetValuesIn(BaseModel):
    session_id: str = Field(..., description="session id from /intake/start")
    values: Dict[str, Any] = Field(..., description="semantic_name -> value")
    lock: bool = Field(default=True, description="lock fields after setting")

# ===================== CORE: start/chat/json/status =====================
@app.post("/intake/start", tags=["1. start"], summary="Upload PDF & start session")
async def intake_start(
    file: UploadFile = File(...),
    omit_nulls: Optional[bool] = Form(False),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf")

    sid = str(uuid.uuid4())
    pdf_path = os.path.join(DATA_DIR, f"{sid}.pdf")
    json_path = os.path.join(DATA_DIR, f"{sid}.json")

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    plain = extract_plain(pdf_path)
    fields = llm_schema(plain)
    if not fields:
        raise HTTPException(status_code=422, detail="Could not infer fields from this document.")

    name_to_field = {f["name"] : f for f in fields}
    required = [f["name"] for f in fields if f.get("required", True)]
    optional = [f["name"] for f in fields if not f.get("required", True)]

    known: Dict[str, Any] = {}
    invalid: Dict[str, str] = {}
    locked: set[str] = set()
    offered_optional = False

    remaining_required = [n for n in required if (n not in known or known.get(n) in (None, "")) and n not in locked]
    have_required_done = (not remaining_required) and (not invalid)
    if have_required_done and optional and not offered_optional:
        offered_optional = True

    outstanding = len(remaining_required) + len(invalid)
    max_q = 3 if outstanding > 6 else (2 if outstanding > 2 else 1)

    focus_names = list(invalid.keys())[:2]
    if not have_required_done:
        focus_names += [n for n in remaining_required if n not in focus_names][:6]
    else:
        next_opt = [n for n in optional if (n not in known and n not in locked)]
        focus_names += next_opt[:6]
    focus_subset = [name_to_field[n] for n in focus_names if n in name_to_field]

    askable = []
    for fmeta in focus_subset:
        fname = fmeta["name"]
        already = (fname in locked) or (fname in known and known[fname] not in (None, ""))
        if not already:
            askable.append(fmeta)

    last_q = ""
    if askable:
        last_q = llm_plan(known, invalid, askable, max_q, have_required_done, offered_optional, locked=list(locked))

    SESSIONS[sid] = {
        "pdf_path": pdf_path,
        "fields": fields,
        "known": known,
        "invalid": invalid,
        "required": required,
        "optional": optional,
        "locked": locked,
        "offered_optional": offered_optional,
        "last_q": last_q,
        "json_path": json_path,
        "omit_nulls": bool(omit_nulls),
        "created_at": int(time.time()),
    }

    write_simple_json(json_path, fields, known, omit_nulls=bool(omit_nulls))

    return {
        "session_id": sid,
        "bot_message": last_q or "Hi! I’ll guide you and fill this document. If something doesn’t apply, say “N/A” or “leave blank”.",
        "done": (last_q == ""),
        "progress": progress(fields, known, required),
    }

@app.post("/intake/message", tags=["2. chat"], summary="Chat with the intake bot")
async def intake_message(msg: MessageIn):
    if msg.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if not msg.user_text or not msg.user_text.strip():
        raise HTTPException(status_code=400, detail="user_text is empty")

    sess = SESSIONS[msg.session_id]
    fields = sess["fields"]; known = sess["known"]; invalid = sess["invalid"]
    required = sess["required"]; optional = sess["optional"]
    locked: set[str] = sess["locked"]; offered_optional: bool = sess["offered_optional"]
    last_q = sess["last_q"]; json_path = sess["json_path"]; omit_nulls = sess["omit_nulls"]
    name_to_field = {f["name"]: f for f in fields}

    ex = llm_extract(fields, known, last_q, msg.user_text)

    for fname in ex["leave_blank"]:
        fmeta = name_to_field.get(fname)
        if fmeta and not fmeta.get("required", True):
            known[fname] = None
            locked.add(fname)

    for fname in ex["confirm"]:
        if fname in ex["proposed"]:
            v = ex["proposed"][fname]
            fmeta = name_to_field.get(fname, {})
            ftype = (fmeta.get("type","text") or "text").lower()
            if is_na(v) and ftype in ("boolean","checkbox"):
                known[fname] = "No"
            elif is_na(v):
                known[fname] = None
            else:
                normv, _ = validate(fmeta, v)
                known[fname] = normv if normv is not None else v
            locked.add(fname)

    for fname in ex["reject"]:
        fmeta = name_to_field.get(fname)
        if fmeta and fmeta.get("required", True):
            invalid[fname] = "Please provide the correct value."

    new_confirms, new_errors = {}, {}
    for k, v in ex["values"].items():
        if k not in name_to_field:
            continue
        fmeta = name_to_field[k]
        ftype = (fmeta.get("type","text") or "text").lower()
        if is_na(v):
            if ftype in ("boolean","checkbox"):
                known[k] = "No"
            else:
                known[k] = None
            locked.add(k)
            continue
        norm, err = validate(fmeta, v)
        if err:
            if fmeta.get("required", True):
                new_errors[k] = err
        else:
            new_confirms[k] = norm

    for k, v in new_confirms.items():
        known[k] = v
        locked.add(k)

    invalid = new_errors
    sess["invalid"] = invalid

    to_confirm_now = {}
    to_confirm_now.update({k:v for k,v in ex["proposed"].items() if k in ex["confirm"]})
    to_confirm_now.update(new_confirms)
    confirmation = llm_confirm(to_confirm_now) if to_confirm_now else ""

    _ = write_simple_json(json_path, fields, known, omit_nulls=omit_nulls)

    remaining_required = [n for n in required if (n not in known or known[n] in (None,"")) and n not in locked]
    have_required_done = (not remaining_required) and (not invalid)

    if have_required_done and optional and not offered_optional:
        offered_optional = True
        sess["offered_optional"] = True

    all_optional_done = (not optional) or all((n in known or n in locked) for n in optional)
    if have_required_done and all_optional_done:
        sess["last_q"] = ""
        return {
            "confirmation": confirmation or None,
            "bot_message": None,
            "done": True,
            "progress": progress(fields, known, required),
            "json": build_simple_map(fields, known),
        }

    outstanding = len(remaining_required) + len(invalid)
    max_q = 3 if outstanding > 6 else (2 if outstanding > 2 else 1)

    focus_names = list(invalid.keys())[:2]
    if not have_required_done:
        focus_names += [n for n in remaining_required if n not in focus_names][:6]
    else:
        next_opt = [n for n in optional if (n not in known and n not in locked)]
        focus_names += next_opt[:6]
    focus_subset = [name_to_field[n] for n in focus_names if n in name_to_field]

    askable = []
    for fmeta in focus_subset:
        fname = fmeta["name"]
        already = (fname in locked) or (fname in known and known[fname] not in (None, ""))
        if not already:
            askable.append(fmeta)

    next_q = "Could you provide any missing required details?" if not askable else llm_plan(
        known, invalid, askable, max_q, have_required_done, offered_optional, locked=list(locked))

    sess["last_q"] = next_q
    return {
        "confirmation": confirmation or None,
        "bot_message": next_q,
        "done": False,
        "progress": progress(fields, known, required),
    }

@app.post("/intake/set-values", tags=["2. chat"], summary="Set multiple values directly (validation + N/A handling)")
async def set_values(body: SetValuesIn):
    if body.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    sess = SESSIONS[body.session_id]
    fields = sess["fields"]; known = sess["known"]; invalid = sess["invalid"]
    required = sess["required"]; locked: set[str] = sess["locked"]
    json_path = sess["json_path"]; omit_nulls = sess["omit_nulls"]
    name_to_field = {f["name"]: f for f in fields}

    new_confirms, new_errors = {}, {}
    for k, v in body.values.items():
        if k not in name_to_field:
            continue
        fmeta = name_to_field[k]
        ftype = (fmeta.get("type","text") or "text").lower()
        if is_na(v):
            known[k] = "No" if ftype in ("boolean","checkbox") else None
            if body.lock: locked.add(k)
            continue
        norm, err = validate(fmeta, v)
        if err:
            if fmeta.get("required", True):
                new_errors[k] = err
        else:
            new_confirms[k] = norm

    for k, v in new_confirms.items():
        known[k] = v
        if body.lock: locked.add(k)

    invalid = new_errors
    sess["invalid"] = invalid

    _ = write_simple_json(json_path, fields, known, omit_nulls=omit_nulls)

    remaining_required = [n for n in required if (n not in known or known[n] in (None,"")) and n not in locked]
    have_required_done = (not remaining_required) and (not invalid)

    next_q = sess.get("last_q") or ""
    if have_required_done:
        sess["last_q"] = ""
    else:
        outstanding = len(remaining_required) + len(invalid)
        max_q = 3 if outstanding > 6 else (2 if outstanding > 2 else 1)
        focus_names = list(invalid.keys())[:2] + remaining_required[:6]
        focus_subset = [name_to_field[n] for n in focus_names if n in name_to_field]
        askable = [f for f in focus_subset if not (f["name"] in locked or (f["name"] in known and known[f["name"]] not in (None,"")))]
        if askable:
            next_q = llm_plan(known, invalid, askable, max_q, have_required_done, sess["offered_optional"], locked=list(locked))
            sess["last_q"] = next_q

    return {
        "set_ok": True,
        "invalid": invalid,
        "progress": progress(fields, known, required),
        "next_question": next_q or None,
        "json": build_simple_map(fields, known),
    }

@app.get("/intake/json", tags=["3. export"], summary="Get collected JSON (simple {field: value})")
async def intake_json(
    session_id: str = Query(...),
    omit_nulls: Optional[bool] = Query(None),
):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    sess = SESSIONS[session_id]
    fields = sess["fields"]; known = sess["known"]
    use_omit = sess["omit_nulls"] if omit_nulls is None else bool(omit_nulls)
    return JSONResponse(build_simple_map(fields, known, omit_nulls=use_omit))

@app.get("/intake/status", tags=["debug"], summary="Status & progress")
async def intake_status(session_id: str = Query(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    sess = SESSIONS[session_id]
    fields = sess["fields"]; known = sess["known"]
    invalid = sess["invalid"]; required = sess["required"]; optional = sess["optional"]
    locked = list(sess["locked"])
    missing_required = [n for n in required if (n not in known or known[n] in (None,"")) and (n not in sess["locked"])]
    missing_optional = [n for n in optional if (n not in known or known[n] in (None,"")) and (n not in sess["locked"])]
    filled = {k:v for k,v in known.items() if v not in (None,"")}
    return {
        "session_id": session_id,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "invalid": invalid,
        "locked": locked,
        "filled_count": len(filled),
        "filled_keys": sorted(filled.keys()),
        "progress": progress(fields, known, required),
        "last_question": sess.get("last_q") or ""
    }

# ===================== HEALTH =====================
@app.get("/health", tags=["0. config"], summary="Health check")
async def health():
    return {"status": "ok", "time": int(time.time())}
