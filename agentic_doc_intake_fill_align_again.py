# agentic_doc_intake_fill_natural_align.py
import os, sys, json, re, time, math
from datetime import date
from dateutil.parser import parse as parse_date
import fitz  # PyMuPDF
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
import httpx

# =========================
# CONFIG
# =========================
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
DEBUG = False  # True draws outline boxes to help visualize placement

# Rendering config (fallback drawing, not used for AcroForm widgets)
TEXT_FONT = "helv"
TEXT_SIZE = 10
LINE_LEADING = 12
MAX_CHARS_PER_LINE = 72

# Layout heuristics for blank detection
Y_TOL = 6
UNDERLINE_MIN_LEN = 6
DOTLINE_MIN_LEN = 6
FALLBACK_GAP = 16
RIGHT_MARGIN_PAD = 24
MIN_HLINE_LEN = 60
HLINE_Y_TOL = 5
BLANK_HEIGHT = 16
BLANK_TOP_PAD = 2
LEFT_INSET = 2
TOP_INSET = 2

# Date rules (by substring in field name/label)
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

NA_TOKENS = {
    "n/a","na","not applicable","none","no number","blank","skip","leave blank","leave it blank","leave as blank"
}

def is_na(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return re.sub(r"\s+"," ", v.strip().lower()) in NA_TOKENS
    # e.g. user says False for an optional checkbox to leave blank.
    return False

def _norm(t):
    # handle non-strings (e.g., True/False) gracefully
    if isinstance(t, str):
        return re.sub(r"\s+", " ", t.strip().lower())
    return re.sub(r"\s+", " ", str(t).strip().lower())

# =========================
# OPENAI CLIENT + RETRIES
# =========================
client = OpenAI(timeout=60)

def call_openai(fn, max_retries=5, backoff=2):
    last_err = None
    for a in range(1, max_retries + 1):
        try:
            return fn()
        except (APIConnectionError, APITimeoutError, httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError, RateLimitError) as e:
            last_err = e
            sleep_s = backoff ** a
            print(f"Network hiccup (attempt {a}/{max_retries}): {e}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)
    raise last_err

# =========================
# PDF TEXT + POSITIONS
# =========================
def extract_pages_and_plain(pdf_path):
    doc = fitz.open(pdf_path)
    pages, plain_chunks = [], []
    for pi in range(len(doc)):
        page = doc[pi]
        spans = []
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    t = s.get("text","")
                    if not t or not t.strip(): 
                        continue
                    bbox = list(fitz.Rect(s["bbox"]))
                    spans.append({"text": t, "bbox": bbox})
        pages.append({
            "spans": spans,
            "width": page.rect.width,
            "height": page.rect.height,
            "drawings": page.get_drawings(),  # vector lines/rectangles
        })
        plain_chunks.append(f"[PAGE {pi+1}]\n" + " ".join([s["text"] for s in spans]))
    doc.close()
    return pages, "\n\n".join(plain_chunks)

def group_spans_by_line(spans):
    lines = []
    for s in spans:
        y_mid = (s["bbox"][1] + s["bbox"][3]) / 2.0
        placed = False
        for line in lines:
            if abs(y_mid - line["y_mid"]) <= Y_TOL:
                line["spans"].append(s)
                line["y_mid"] = (line["y_mid"] * line["count"] + y_mid) / (line["count"] + 1)
                line["count"] += 1
                placed = True
                break
        if not placed:
            lines.append({"y_mid": y_mid, "count": 1, "spans": [s]})
    for ln in lines:
        ln["spans"].sort(key=lambda sp: sp["bbox"][0])
    lines.sort(key=lambda ln: ln["y_mid"])
    return lines

def find_label_line(lines, target_norm):
    for ln in lines:
        line_text = " ".join(s["text"] for s in ln["spans"])
        if target_norm in _norm(line_text):
            return ln
    return None

# --- replace your detect_horizontal_line_blank with this ---
def detect_horizontal_line_blank(page, label_right, label_y_mid, y_window=HLINE_Y_TOL):
    """
    Look for a horizontal vector line at approximately the given y, and to the right of the label.
    """
    for d in page["drawings"]:
        pts = d.get("points")
        if not pts or len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i+1]
            # horizontal line?
            if abs(y0 - y1) <= 0.6:
                length = abs(x1 - x0)
                if length >= MIN_HLINE_LEN:
                    y_line = (y0 + y1) / 2.0
                    if abs(y_line - label_y_mid) <= y_window:
                        x_left, x_right = min(x0, x1), max(x0, x1)
                        if x_right > label_right + 4:
                            y_top = y_line - BLANK_TOP_PAD
                            y_bot = y_top + BLANK_HEIGHT
                            return [x_left, y_top, x_right, y_bot]
    return None


def _line_has_underline_spans(spans, x_min, min_len=UNDERLINE_MIN_LEN):
    for s in spans:
        txt = s["text"]
        if not txt:
            continue
        if s["bbox"][0] < x_min - 0.5:
            continue
        if ("_" * min_len) in txt or ("." * DOTLINE_MIN_LEN) in txt:
            return [s["bbox"][0], s["bbox"][1], s["bbox"][2], s["bbox"][3]]
    return None


def detect_blank_region_near_label(lines, line_index, label_span_idx, page, label_right):
    """
    Try, in this order:
      1) Horizontal rule on the same line as the label (right of it)
      2) Underline/dotted spans on the same line (right of it)
      3) Horizontal rule within ~1 line below
      4) Underline/dotted spans in the next 1-2 lines below
      5) Fallback gap region to the right (same line)
      6) Very conservative right-side region toward page margin
    """
    line = lines[line_index]
    spans = line["spans"]
    y_top = min(s["bbox"][1] for s in spans)
    y_bot = max(s["bbox"][3] for s in spans)

    # 1) horizontal rule on this line
    h_blank = detect_horizontal_line_blank(page, label_right, line["y_mid"], y_window=HLINE_Y_TOL)
    if h_blank:
        return h_blank

    # 2) underline/dots on this line (to the right of label)
    right_spans = [s for s in spans if s["bbox"][0] >= label_right - 0.5]
    ul = _line_has_underline_spans(right_spans, label_right)
    if ul:
        return [ul[0], y_top, ul[2], y_bot]

    # 3) horizontal rule on the next line down (sometimes the blank is below)
    if line_index + 1 < len(lines):
        below = lines[line_index + 1]
        h_blank_below = detect_horizontal_line_blank(page, label_right, below["y_mid"], y_window=HLINE_Y_TOL + 6)
        if h_blank_below:
            return h_blank_below

    # 4) underline/dots in the next 1–2 lines below
    for j in (line_index + 1, line_index + 2):
        if 0 <= j < len(lines):
            l2 = lines[j]
            ul2 = _line_has_underline_spans(l2["spans"], label_right)
            if ul2:
                y_top2 = min(s["bbox"][1] for s in l2["spans"])
                y_bot2 = max(s["bbox"][3] for s in l2["spans"])
                return [ul2[0], y_top2, ul2[2], y_bot2]

    # 5) gap to next span on the same line
    if right_spans:
        nxt = right_spans[0]
        x0 = label_right + FALLBACK_GAP
        x1 = max(x0 + 120, nxt["bbox"][0] - 3)
        return [x0, y_top, x1, y_bot]

    # 6) conservative right-side region
    pw = page["width"]
    x0 = label_right + FALLBACK_GAP
    x1 = max(x0 + 140, pw - RIGHT_MARGIN_PAD)
    return [x0, y_top, x1, y_bot]

# --- replace your detect_blank_region_on_line with this (now calls the new helper) ---
def detect_blank_region_on_line(line, label_span_idx, page, all_lines, line_index):
    spans = line["spans"]
    label_right = spans[label_span_idx]["bbox"][2] if label_span_idx is not None else min(s["bbox"][0] for s in spans)

    blank_bbox = detect_blank_region_near_label(all_lines, line_index, label_span_idx, page, label_right)
    if blank_bbox:
        x0, y0, x1, y1 = blank_bbox
        x1 = min(x1, page["width"] - RIGHT_MARGIN_PAD)
        return [x0, y0, x1, y1]
    return None

# --- replace your find_anchors_and_blanks with this (passes all_lines & line_index) ---
def find_anchors_and_blanks(pages, fields):
    placements = {}
    for f in fields:
        target = _norm(f.get("anchor_phrase") or f.get("label") or f["name"])
        page_hint = f.get("page")
        cand_pages = range(len(pages)) if not page_hint else [max(0, min(page_hint-1, len(pages)-1))]

        found = None
        for p in cand_pages:
            spans = pages[p]["spans"]
            lines = group_spans_by_line(spans)
            ln = None
            ln_idx = -1
            for idx, L in enumerate(lines):
                line_text = " ".join(s["text"] for s in L["spans"])
                if target in _norm(line_text):
                    ln = L
                    ln_idx = idx
                    break
            if not ln:
                continue

            label_span_idx = None
            for i, sp in enumerate(ln["spans"]):
                if target in _norm(sp["text"]):
                    label_span_idx = i
                    break

            if label_span_idx is not None:
                anchor_bbox = ln["spans"][label_span_idx]["bbox"]
            else:
                xs = [s["bbox"][0] for s in ln["spans"]]
                xe = [s["bbox"][2] for s in ln["spans"]]
                ys = [s["bbox"][1] for s in ln["spans"]]
                ye = [s["bbox"][3] for s in ln["spans"]]
                anchor_bbox = [min(xs), min(ys), max(xe), max(ye)]

            blank_bbox = detect_blank_region_on_line(ln, label_span_idx, pages[p], lines, ln_idx)
            found = {"page": p+1, "anchor_bbox": anchor_bbox, "blank_bbox": blank_bbox}
            break

        if found:
            placements[f["name"]] = found

    return placements


def draw_debug_boxes(doc, placements, color=(1,0,0)):
    for name, info in placements.items():
        page = doc[info["page"] - 1]
        if info.get("anchor_bbox"):
            page.draw_rect(info["anchor_bbox"], color=color, width=0.6)
        if info.get("blank_bbox"):
            page.draw_rect(info["blank_bbox"], color=color, width=0.6)

# =========================
# ACROFORM WIDGET MAPPING
# =========================
def page_text_left_of_rect(page_spans, rect, y_tol=9, max_left_dx=250):
    """Find the closest text to the left of a rect on approximately the same line."""
    rx0, ry0, rx1, ry1 = rect
    y_mid = (ry0 + ry1) / 2.0
    best = None
    best_dx = 1e9
    for s in page_spans:
        sx0, sy0, sx1, sy1 = s["bbox"]
        s_mid = (sy0 + sy1) / 2.0
        if abs(s_mid - y_mid) <= y_tol and sx1 <= rx0:  # to the left
            dx = rx0 - sx1
            if 0 <= dx <= max_left_dx and dx < best_dx:
                best_dx = dx
                best = s
    return best

def get_widget_label_text(pages_spans, page_index, widget_rect):
    s = page_text_left_of_rect(pages_spans[page_index]["spans"], widget_rect)
    return s["text"] if s else ""

def collect_widgets(doc):
    """Return widgets with geometry and label text."""
    widgets = []
    for p in range(len(doc)):
        page = doc[p]
        page_widgets = page.widgets() or []
        for w in page_widgets:
            rect = list(w.rect)
            widgets.append({
                "page": p+1,
                "widget": w,
                "rect": rect,
                "type": w.field_type,          # e.g., TEXT, CHECKBOX, etc.
                "name": w.field_name or "",
            })
    return widgets

def map_fields_to_widgets(pages_spans, fields, widgets, placements):
    """
    Try to map our semantic fields to AcroForm widgets.
    Strategy:
      1) If we already located a label's anchor/blank (placements),
         pick the widget on the same page with the closest vertical center to that blank/anchor.
      2) Else match by nearest left-of-widget label text vs field label.
    """
    # Build label text per widget
    for w in widgets:
        pidx = w["page"] - 1
        w["label_text"] = get_widget_label_text(pages_spans, pidx, w["rect"])

    # Helper: pick widget on same page with closest y-mid to target bbox
    def widget_near_bbox(page_no, bbox):
        tx0, ty0, tx1, ty1 = bbox
        tmid = (ty0 + ty1) / 2.0
        cands = [w for w in widgets if w["page"] == page_no]
        if not cands:
            return None
        best, best_d = None, 1e9
        for w in cands:
            rx0, ry0, rx1, ry1 = w["rect"]
            wmid = (ry0 + ry1) / 2.0
            d = abs(wmid - tmid)
            if d < best_d:
                best, best_d = w, d
        return best

    mapped = {}   # field_name -> widget
    for f in fields:
        fname = f["name"]
        # Prefer placements (from label search)
        if fname in placements:
            info = placements[fname]
            bbox = info.get("blank_bbox") or info.get("anchor_bbox")
            w = widget_near_bbox(info["page"], bbox)
            if w:
                mapped[fname] = w
                continue

        # Fallback: label similarity to widget label_text
        f_label_norm = _norm(f.get("label", "") or f.get("anchor_phrase","") or f["name"])
        best, best_score = None, 0
        for w in widgets:
            wlabel = _norm(w.get("label_text",""))
            if not wlabel: 
                continue
            ftoks = set(f_label_norm.split())
            wtoks = set(wlabel.split())
            score = len(ftoks & wtoks)
            if score > best_score:
                best, best_score = w, score
        if best:
            mapped[fname] = best

    return mapped

def fill_widgets(doc, widget_map, answers):
    """
    Fill AcroForm widgets exactly (no drawing). This keeps alignment perfect.
    """
    filled_any = False
    for fname, w in widget_map.items():
        val = answers.get(fname, None)
        if val in (None, ""):
            continue
        wid = w["widget"]
        wtype = w["type"]
        if isinstance(val, bool):
            val = "Yes" if val else "No"
        vnorm = _norm(val)
        try:
            if wtype == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                wid.field_value = "Yes" if vnorm in ("yes","y","true","1","checked") else "Off"
                wid.update()
                filled_any = True
            elif wtype == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
                wid.field_value = "Yes" if vnorm in ("yes","y","true","1","checked") else "Off"
                wid.update()
                filled_any = True
            else:
                wid.field_value = str(val)
                wid.update()
                filled_any = True
        except Exception:
            continue
    return filled_any

# =========================
# FALLBACK DRAWING (no AcroForm)
# =========================
def wrap_text(s, width):
    words, lines, cur = s.split(), [], []
    for w in words:
        tst = (" ".join(cur+[w])).strip()
        if len(tst) <= width: cur.append(w)
        else: lines.append(" ".join(cur)); cur = [w]
    if cur: lines.append(" ".join(cur))
    return lines or [""]

# --- replace your render_answers_draw with this version ---
def render_answers_draw(pdf_in, pdf_out, placements, answers, fields_meta=None):
    """
    Draw text into detected blanks. For checkbox/boolean fields without widgets,
    draw a centered checkmark instead of the text 'Yes'.
    """
    fields_meta = fields_meta or []
    meta_by_name = {f["name"]: f for f in fields_meta}

    doc = fitz.open(pdf_in)
    if DEBUG: draw_debug_boxes(doc, placements, color=(0,0,1))

    for name, info in placements.items():
        val = answers.get(name)
        if val in (None, ""):
            continue

        page = doc[info["page"] - 1]
        target = info.get("blank_bbox") or info.get("anchor_bbox")
        if not target:
            continue

        x0, y0, x1, y1 = target
        fmeta = meta_by_name.get(name, {})
        ftype = (fmeta.get("type") or "text").lower()

        # Handle checkboxes/booleans specially
        if ftype in ("boolean", "checkbox"):
            vnorm = _norm(val)
            is_yes = vnorm in ("yes","y","true","1","checked")
            if is_yes:
                # Draw a crisp checkmark centered in the box
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                # Choose a font size that fits the box nicely
                box_h = max(8, (y1 - y0))
                fs = min(14, max(10, box_h - 2))
                page.insert_text((cx, cy), "✓",
                                 fontname=TEXT_FONT,  # helv contains ✓ in modern viewers
                                 fontsize=fs,
                                 color=(0,0,0),
                                 render_mode=0,
                                 align=1)   # center on the point
            # If No/unchecked: draw nothing (leave the box empty)
            continue

        # Normal text fields
        x = x0 + LEFT_INSET
        baseline = y0 + TOP_INSET + (y1 - y0) * 0.65
        text = str(val)
        for i, line in enumerate(wrap_text(text, MAX_CHARS_PER_LINE)):
            page.insert_text((x, baseline - i*LINE_LEADING), line,
                             fontname=TEXT_FONT, fontsize=TEXT_SIZE, fill=(0,0,0))

    doc.save(pdf_out)
    doc.close()


# =========================
# VALIDATION
# =========================
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^[+()\- \d]{7,}$")

def pick_date_policy(field_name, field_label):
    key = _norm((field_name or "") + " " + (field_label or ""))
    for hint, rule in DATE_RULES.items():
        if hint in key:
            return rule
    return {"future_ok": False, "allow_today": True}

def norm_date_for_field(field_name, field_label, value):
    try:
        dt = parse_date(str(value), dayfirst=True, yearfirst=False)
        today = date.today()
        pol = pick_date_policy(field_name, field_label)
        if dt.date() > today and not pol["future_ok"]:
            return None, "That date appears to be in the future; please confirm or provide a past/today date."
        if dt.date() == today and not pol["allow_today"]:
            return None, "Please provide a date that is not today."
        return dt.strftime("%Y-%m-%d"), None
    except Exception:
        return None, "I couldn’t parse that date—please use dd/mm/yyyy or mm/dd/yyyy."

def validate(field_meta, val):
    # Explicit N/A-like answers: intentionally blank (caller will lock it)
    if is_na(val):
        return None, None

    if val is None:
        return None, "Missing value"
    if isinstance(val, str) and not val.strip():
        return None, "Missing value"

    ftype = field_meta.get("type", "text")
    label = field_meta.get("label", "")
    name  = field_meta.get("name", "")

    if ftype == "date":
        return norm_date_for_field(name, label, val)
    if ftype == "email":
        v = str(val).strip()
        return (v, None) if EMAIL_RE.match(v) else (None, "That email doesn’t look valid—mind checking?")
    if ftype == "phone":
        v = re.sub(r"\s+"," ", str(val).strip())
        return (v, None) if PHONE_RE.match(v) else (None, "Could you add a few more digits (and country code if needed)?")
    if ftype in ("boolean","checkbox"):
        vv = _norm(val)
        if vv in ("yes","y","true","1","checked"):   return "Yes", None
        if vv in ("no","n","false","0","unchecked"): return "No", None
        return None, "Please answer yes or no."
    return (str(val).strip() or None), None

# =========================
# LLM PROMPTS (unchanged convo flow)
# =========================
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
- Include all likely user-supplied fields (no arbitrary limits).
- Use 'boolean' for yes/no; 'checkbox' for visible checkboxes.
- If optional (“if applicable”), set required=false.
- Prefer concise, stable names (snake_case).
"""

PLAN_SYS_TEMPLATE = """You are a warm, efficient intake agent.
Goal: naturally collect the remaining details for this document.

HARD RULES:
- Ask ONLY about the provided FOCUS_FIELDS. Do NOT ask about anything else.
- Do NOT ask about fields that appear in KNOWN or in LOCKED unless they also appear in NEEDS_CLARIFICATION.
- Treat any value present in KNOWN (including “Yes”/“No”) as final unless that field is explicitly listed in NEEDS_CLARIFICATION.
- If the last user reply already answered a binary/yes-no field (e.g., “No travel plans”), do not ask that field again.

Guidelines:
- Ask at most {max_q} short questions this turn (1–3), phrased like a human helper.
- Group related items gracefully (e.g., "When did you start, and what’s your job title?").
- DO NOT dump a checklist of all remaining fields; progress gradually.
- Acknowledge prior answers; avoid repeats unless clarifying.
- Respect N/A (or "leave blank") and move on.
- If a prior value seemed invalid (e.g., future DOB), politely confirm once and accept the user’s clarification.
- After all required fields are done, explicitly ask: “There are some optional items (e.g., X, Y). Want to add them now?” If no, finish. If yes, gather them in small clusters.

Write only conversational text, no JSON/system notes.
"""


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# UPDATED: Intent-aware extractor (confirmation / rejection / leave blank)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
EXTRACT_SYS = """You extract structured updates from the user's free-text reply.

Inputs you receive:
- SCHEMA: array of fields (with 'name', 'label', 'type', 'required', 'anchor_phrase', 'page').
- KNOWN:  map of already known values (may be None for intentionally blank).
- LAST_QUESTION: the last bot message; it may contain proposed values such as:
  - Would you like to use "Smith Dave" as the H-1 employee name?
  - Keep Phone Number as "555-123-4567"?
  - Use "N/A" for Current Visa Type?

Your job:
1) Do NOT infer or copy values that the user did not explicitly provide.
2) Detect high-level intent even when phrased broadly:
   - Confirmation (agree/approve/accept/ok/sounds good, etc.).
   - Rejection (disagree/that’s wrong/no/please change/etc.).
   - Leave blank / skip (N/A, not applicable, leave blank, skip, etc.).
3) Only resolve "same as above" if the user explicitly writes that phrase and the referenced field has a concrete value in KNOWN.
4) Ignore empty/whitespace replies.

Return STRICT JSON (omit empty keys):
{
  "values": { "<field_name>": "<explicit_value_from_user>", ... },
  "confirm": ["<field_name>", ...],
  "reject": ["<field_name>", ...],
  "leave_blank": ["<field_name>", ...],
  "proposed": { "<field_name>": "<value_from_last_question>", ... }
}
"""

CONFIRM_SYS = """In 1–3 short lines, politely confirm the most recently added or corrected key values (plain text, no JSON)."""

def llm_schema(plain):
    msgs=[{"role":"system","content":SCHEMA_SYS},
          {"role":"user","content":f"DOCUMENT_TEXT (first 10k chars):\n{plain[:10000]}"}]
    r = call_openai(lambda: client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, messages=msgs, response_format={"type":"json_object"}
    ))
    try: return json.loads(r.choices[0].message.content).get("fields", [])
    except: return []

def llm_plan(known, invalid, focus_subset, max_q, have_required_done, offered_optional, locked=None):
    if locked is None:
        locked = []
    plan_sys = PLAN_SYS_TEMPLATE.format(max_q=max_q)
    intent = {"have_required_done": have_required_done, "have_offered_optional": offered_optional}
    msgs=[{"role":"system","content":plan_sys},
          {"role":"user","content":(
              "KNOWN:\n" + json.dumps(known,ensure_ascii=False) +
              "\n\nLOCKED:\n" + json.dumps(list(locked),ensure_ascii=False) +
              "\n\nNEEDS_CLARIFICATION:\n" + json.dumps(invalid,ensure_ascii=False) +
              "\n\nFOCUS_FIELDS:\n" + json.dumps(focus_subset,ensure_ascii=False) +
              "\n\nINTENT:\n" + json.dumps(intent)
          )}]
    r = call_openai(lambda: client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, messages=msgs
    ))
    return r.choices[0].message.content.strip()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# UPDATED: llm_extract returns intents + explicit values
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def llm_extract(schema, known, last_q, user_reply):
    if not user_reply or not str(user_reply).strip():
        return {"values":{}, "confirm":[], "reject":[], "leave_blank":[], "proposed":{}}
    msgs=[{"role":"system","content":EXTRACT_SYS},
          {"role":"user","content":f"SCHEMA:{json.dumps(schema,ensure_ascii=False)}\nKNOWN:{json.dumps(known,ensure_ascii=False)}\nLAST_QUESTION:{last_q}\nREPLY:{user_reply}"}]
    r = call_openai(lambda: client.chat.completions.create(
        model=MODEL, temperature=0, messages=msgs, response_format={"type":"json_object"}
    ))
    try:
        raw = json.loads(r.choices[0].message.content)
    except Exception:
        raw = {}
    # normalize structure
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
    r = call_openai(lambda: client.chat.completions.create(
        model=MODEL, temperature=0.2, messages=msgs
    ))
    return r.choices[0].message.content.strip()

# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) < 3:
        print("Usage: python agentic_doc_intake_fill_natural_align.py input.pdf output.pdf")
        sys.exit(1)
    pdf_in, pdf_out = sys.argv[1], sys.argv[2]
    if not os.path.exists(pdf_in):
        print("File not found:", pdf_in); sys.exit(1)

    # 1) Parse doc text to propose fields
    pages, plain = extract_pages_and_plain(pdf_in)
    fields = llm_schema(plain)
    if not fields:
        print("I couldn’t infer fields from this document automatically."); sys.exit(1)

    name_to_field = {f["name"]: f for f in fields}
    required = [f["name"] for f in fields if f.get("required", True)]
    optional = [f["name"] for f in fields if not f.get("required", True)]
    placements = find_anchors_and_blanks(pages, fields)

    known, locked, invalid = {}, set(), {}
    last_q = ""
    offered_optional = False

    print("Bot: Hi! I’ll guide you and fill this document. If something doesn’t apply, just say “N/A” or “leave blank”.")
    while True:
        remaining_required = [n for n in required if (n not in known or known[n] in (None,"")) and n not in locked]
        have_required_done = (not remaining_required) and (not invalid)

        if have_required_done and optional and not offered_optional:
            offered_optional = True  # planner will start offering optional

        all_optional_done = (not optional) or all((n in known or n in locked) for n in optional)
        if have_required_done and all_optional_done:
            break

        outstanding = len(remaining_required) + len(invalid)
        max_q = 3 if outstanding > 6 else (2 if outstanding > 2 else 1)

        focus_names = list(invalid.keys())[:2]
        if not have_required_done:
            focus_names += [n for n in remaining_required if n not in focus_names][:6]
        else:
            next_opt = [n for n in optional if (n not in known and n not in locked)]
            focus_names += next_opt[:6]
        focus_subset = [name_to_field[n] for n in focus_names if n in name_to_field]

        # >>> HARD GUARD: don't ask about already-answered or locked fields
        askable = []
        for f in focus_subset:
            fname = f["name"]
            already = (fname in locked) or (fname in known and known[fname] not in (None, ""))
            if not already:
                askable.append(f)

        # If nothing is left to ask (planner would just repeat), recompute next loop iteration
        if not askable:
            # If all required are done, we'll eventually break; otherwise the loop
            # will calculate a new focus set on the next iteration.
            continue

        # Use askable instead of focus_subset from here on:
        focus_subset = askable


        last_q = llm_plan(known, invalid, focus_subset, max_q, have_required_done, offered_optional, locked=list(locked))

        print("\nBot:", last_q)
        user = input("You: ")
        if user.lower().strip() in ("quit","exit"):
            sys.exit(0)

        # >>> UPDATED: intent-aware extraction
        ex = llm_extract(fields, known, last_q, user)

        # 1) Handle "leave blank" (only lock if optional)
        for fname in ex["leave_blank"]:
            fmeta = name_to_field.get(fname)
            if fmeta and not fmeta.get("required", True):
                known[fname] = None
                locked.add(fname)

        # 2) Handle "confirm" (only if there is a proposed value in last question)
        for fname in ex["confirm"]:
            if fname in ex["proposed"]:
                known[fname] = ex["proposed"][fname]
                locked.add(fname)

        # 3) Handle "reject" (keep required fields pending / invalid)
        for fname in ex["reject"]:
            fmeta = name_to_field.get(fname)
            if fmeta and fmeta.get("required", True):
                invalid[fname] = "Please provide the correct value."

        # 4) Handle explicit values the user actually provided
        new_confirms, new_errors = {}, {}
        for k, v in ex["values"].items():
            if k not in name_to_field:
                continue
            if is_na(v):
                known[k] = None
                locked.add(k)
                continue
            norm, err = validate(name_to_field[k], v)
            if err:
                if name_to_field[k].get("required", True):
                    new_errors[k] = err
            else:
                new_confirms[k] = norm

        # Merge confirmed values
        for k, v in new_confirms.items():
            known[k] = v
            locked.add(k)

        # Replace invalid with fresh errors from this turn
        invalid = new_errors

        # Quick confirmation line
        to_confirm_now = {}
        to_confirm_now.update({k:v for k,v in ex["proposed"].items() if k in ex["confirm"]})
        to_confirm_now.update(new_confirms)
        if to_confirm_now:
            conf = llm_confirm(to_confirm_now)
            if conf:
                print("\nBot (quick check):", conf)

    # 2) Fill the PDF
    # Try AcroForm first (perfect placement), otherwise fallback to drawing inside detected blanks
    doc = fitz.open(pdf_in)
    widgets = collect_widgets(doc)
    doc.close()

    if widgets:
        widget_map = map_fields_to_widgets(pages, fields, widgets, placements)
        doc = fitz.open(pdf_in)
        filled = fill_widgets(doc, widget_map, known)
        doc.save(pdf_out, deflate=True)
        doc.close()
        if filled:
            print(f"\n✅ Done! Filled AcroForm fields → {pdf_out}")
        else:
            render_answers_draw(pdf_in, pdf_out, placements, known, fields_meta=fields)
            print(f"\n✅ Done! Drew text into blanks → {pdf_out}")
    else:
        render_answers_draw(pdf_in, pdf_out, placements, known, fields_meta=fields)
        print(f"\n✅ Done! Drew text into blanks → {pdf_out}")

if __name__ == "__main__":
    main()