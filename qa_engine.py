import json
from ollama_utils import ask_llama
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------- NEW: product search deps -----------------------------
import os
import re
import pandas as pd
from typing import List, Tuple

PRODUCT_CSV_PATH = "Product List.csv"
MAX_PRODUCTS_TO_SHOW = 5
# ------------------------------------------------------------------------------------

# Load data from JSON (เดิม)
with open("output_data.json", encoding="utf-8") as f:
    docs = json.load(f)

# Improved Token Matching
def tokenize_and_clean(text: str):
    return word_tokenize(text.lower(), engine='newmm')

# Use TF-IDF for more context-aware document retrieval (เดิม)
def find_best_context(question: str):
    question_words = tokenize_and_clean(question)
    corpus = [entry.get("HTML", "") + " " + entry.get("Header", "") + " " + entry.get("Tag", "") for entry in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    question_vec = vectorizer.transform([" ".join(question_words)])
    similarities = tfidf_matrix * question_vec.T
    scores = similarities.toarray().flatten()
    best_match_idx = scores.argmax()
    if scores[best_match_idx] == 0:
        return None
    return docs[best_match_idx]

# Combine metadata into the context to enhance response generation (เดิม)
def build_prompt(question: str, context_html: str, metadata: dict):
    base = """
กรุณาตอบเป็นภาษาไทยเท่านั้น โดยอ้างอิงจากเนื้อหาของหน้าเว็บไซต์ด้านล่าง
นำข้อมูลทั้งหมดมาตอบเลยและสรุปให้กระชับและเข้าใจง่าย
แต่ถ้ามีคำถามที่มีบริบทเกี่ยวกับ มี/อยู่ไหน สินค้า/บริการ ไหมให้ทำการตอบกลับด้วยลักษณะของการแนะนำสินค้า/บริการที่เกี่ยวข้องกับคำถามนั้นๆ
คำตอบควรมีความละเอียดและชัดเจน ไม่ต้องให้ลิงก์ใดๆในการตอบ

ข้อกำหนด:
- ตอบเฉพาะจากข้อมูลที่มีในบริบทด้านล่างเท่านั้น
- หากสรุป ให้ไม่สั้นจนเกินไปและชัดเจนรายละเอียดครบถ้วน
"""
    meta = f"""[ชื่อหน้าเว็บ]: {metadata.get("NamePage", "")}
[หมวดหมู่/ศูนย์]: {metadata.get("Center", "")}
[หัวข้อ]: {metadata.get("Header", "")}
[แท็ก]: {metadata.get("Tag", "")}"""

    return f"""{base}

คำถาม:
{question}

ข้อมูลสรุปจากหน้าเว็บไซต์:
{meta}

[HTML ของเว็บไซต์]:
{context_html}

คำตอบ:
"""

# ================================== NEW: PRODUCT SEARCH ==================================

def _normalize_product_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ทำชื่อคอลัมน์ให้เป็นมาตรฐาน: ID / ชื่อสินค้า / ศูนย์ / link"""
    rename_map = {}
    lower_map = {c.lower().strip(): c for c in df.columns}

    if "id" in lower_map: rename_map[lower_map["id"]] = "ID"
    for key in ["name", "product", "product name", "ชื่อสินค้า", "รายชื่อสินค้า", "รายการสินค้า"]:
        if key in lower_map: rename_map[lower_map[key]] = "ชื่อสินค้า"; break
    for key in ["center", "ศูนย์", "ศูนย์งาน", "หน่วยงาน"]:
        if key in lower_map: rename_map[lower_map[key]] = "ศูนย์"; break
    for key in ["link", "url", "ลิงก์", "ลิ้งค์"]:
        if key in lower_map: rename_map[lower_map[key]] = "link"; break

    df = df.rename(columns=rename_map)
    for col in ["ID", "ชื่อสินค้า", "ศูนย์", "link"]:
        if col not in df.columns:
            df[col] = ""
    return df[["ID", "ชื่อสินค้า", "ศูนย์", "link"]].copy()

def _load_products() -> pd.DataFrame:
    if not os.path.exists(PRODUCT_CSV_PATH):
        return pd.DataFrame(columns=["ID", "ชื่อสินค้า", "ศูนย์", "link"])
    # รองรับหลาย encoding
    encodings = [None, "utf-8-sig", "cp874", "latin-1"]
    last_err = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(PRODUCT_CSV_PATH, encoding=enc) if enc else pd.read_csv(PRODUCT_CSV_PATH)
            break
        except Exception as e:
            last_err = e
    if df is None:
        return pd.DataFrame(columns=["ID", "ชื่อสินค้า", "ศูนย์", "link"])

    df = _normalize_product_columns(df)
    # ทำคอลัมน์ข้อความเป็น str ป้องกัน NaN
    for c in ["ID", "ชื่อสินค้า", "ศูนย์", "link"]:
        df[c] = df[c].astype(str).fillna("").str.strip()
    return df

_PRODUCTS_DF = _load_products()

def _extract_possible_ids(text: str) -> List[str]:
    """ดึงรูปแบบที่ดูเหมือนรหัสสินค้า (อักษร/ตัวเลข/ขีดล่าง/ขีดกลาง) ยาว >= 3"""
    tokens = re.findall(r"[A-Za-z0-9_-]{3,}", text)
    return [t for t in tokens if not re.search(r"[ก-๙]", t)]

def _tfidf_rank_products(query: str, df: pd.DataFrame, topk: int = 10) -> List[Tuple[int, float]]:
    """จัดอันดับสินค้าด้วย TF-IDF จากคอลัมน์ ID/ชื่อ/ศูนย์/ลิงก์"""
    if df.empty:
        return []
    corpus = (df["ID"].fillna("") + " " +
              df["ชื่อสินค้า"].fillna("") + " " +
              df["ศูนย์"].fillna("") + " " +
              df["link"].fillna("")).tolist()
    vec = TfidfVectorizer()
    X = vec.fit_transform(corpus)
    qv = vec.transform([" ".join(tokenize_and_clean(query))])
    scores = (X * qv.T).toarray().ravel()
    idxs = scores.argsort()[::-1]  # desc
    ranked = [(int(i), float(scores[i])) for i in idxs[:topk] if scores[i] > 0]
    return ranked

def find_related_products(question: str, max_rows: int = MAX_PRODUCTS_TO_SHOW) -> pd.DataFrame:
    """คืน DataFrame สินค้าที่เกี่ยวข้องกับคำถาม (อาจว่างได้)"""
    df = _PRODUCTS_DF
    if df.empty:
        return df

    # 1) ถ้ามี ID ในคำถาม ให้แมตช์ตรงก่อน
    ids = _extract_possible_ids(question.lower())
    exact_hit = pd.DataFrame(columns=df.columns)
    if ids:
        patt = "|".join(map(re.escape, ids))
        exact_hit = df[df["ID"].str.lower().str.contains(patt, na=False)]
        if not exact_hit.empty:
            return exact_hit.head(max_rows).reset_index(drop=True)

    # 2) ไม่พบ ID → ใช้ TF-IDF จัดอันดับความใกล้เคียง
    ranked = _tfidf_rank_products(question, df, topk=max_rows)
    if not ranked:
        return pd.DataFrame(columns=df.columns)
    take_idxs = [i for i, _ in ranked]
    out = df.iloc[take_idxs].copy()
    return out.reset_index(drop=True)

def _mk_link_cell(url: str) -> str:
    u = str(url or "").strip()
    if u.lower().startswith("http://") or u.lower().startswith("https://"):
        return f"[link]({u})"
    return "ไม่พบข้อมูล"

def _products_to_markdown_table(pdf: pd.DataFrame) -> str:
    if pdf.empty:
        return ""
    header = "| ID | ชื่อสินค้า | ศูนย์ | link |\n|---|---|---|---|"
    rows = []
    for _, r in pdf.iterrows():
        rows.append(
            f"| {r['ID']} | {r['ชื่อสินค้า']} | {r['ศูนย์']} | {_mk_link_cell(r['link'])} |"
        )
    return header + "\n" + "\n".join(rows)

# =========================================================================================

# Function to answer the question based on the context (เดิม + แนบสินค้า)
def answer_question(question: str):
    # 1) ระบบเดิม: หา context จากเว็บ TISTR แล้วให้ LLaMA ตอบ
    context = find_best_context(question)
    if not context:
        base_answer = "❌ ไม่พบเนื้อหาที่เกี่ยวข้องในฐานข้อมูล"
        product_df = find_related_products(question)
        product_block = _products_to_markdown_table(product_df)
        if product_block:
            return f"{base_answer}\n\n---\n\n**สินค้า/บริการที่อาจเกี่ยวข้องกับคำถามคุณ:**\n\n{product_block}"
        return base_answer

    html_content = context.get("HTML", "")[:10000]
    prompt = build_prompt(question, html_content, context)
    reply = ask_llama(prompt)
    base_answer = f"{reply.strip()}\n\n🔗 อ้างอิง: {context.get('URL', 'ไม่พบ URL')}"

    # 2) NEW: แนบ “ลิสต์สินค้า” เพิ่มเติม ถ้าดูมีความเกี่ยวข้องกับคำถาม
    product_df = find_related_products(question)
    product_block = _products_to_markdown_table(product_df)
    if product_block:
        base_answer += f"\n\n---\n\n**สินค้า/บริการที่เกี่ยวข้อง:**\n\n{product_block}"

    return base_answer
