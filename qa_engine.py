import json
import difflib
from pythainlp.tokenize import word_tokenize
from ollama_utils import ask_llama

# โหลดข้อมูลจากไฟล์ JSON ที่ถูกดึงมาจากเว็บไซต์
with open("output_data.json", encoding="utf-8") as f:
    docs = json.load(f)

# ฟังก์ชันนี้จะใช้ PyThaiNLP ในการแยกคำ
def preprocess_question(question: str):
    # ใช้ PyThaiNLP เพื่อแยกคำ
    tokenized_words = word_tokenize(question)
    return tokenized_words

# ใช้ HTML เป็นหลักในการค้นหา พร้อมพิจารณา field อื่นๆ รองลงมา
def find_best_context(question: str):
    question_words = preprocess_question(question)  # แยกคำก่อนค้นหา
    scored = []

    for entry in docs:
        html_text = entry.get("HTML", "").lower()
        other_text = " ".join([
            entry.get("Header", ""),
            entry.get("Tag", ""),
            entry.get("NamePage", ""),
            entry.get("Center", "")
        ]).lower()

        html_score = sum(word in html_text for word in question_words) * 3
        other_score = sum(word in other_text for word in question_words) * 1
        total_score = html_score + other_score

        if total_score > 0:
            scored.append((total_score, entry))

    if not scored:
        return None

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]

# ตรวจสอบและแก้ไขคำที่สะกดผิด
def correct_spelling(question: str, keywords: list):
    """ตรวจสอบและแก้ไขคำที่พิมพ์ผิดโดยใช้ difflib"""
    corrected_question = question
    for word in keywords:
        # ใช้ difflib เพื่อเปรียบเทียบคำที่คล้ายกัน
        closest_match = difflib.get_close_matches(word, question.split(), n=1, cutoff=0.8)
        if closest_match:
            corrected_question = corrected_question.replace(closest_match[0], word)
    return corrected_question

# รวม metadata เข้า context + ให้ LLM ตัดสินใจรูปแบบคำตอบเอง (แทน intent)
def build_prompt(question: str, context_html: str, metadata: dict):
    base = """
กรุณาตอบเป็นภาษาไทยเท่านั้น โดยอ้างอิงจากเนื้อหาของหน้าเว็บไซต์ด้านล่าง
ถ้าไม่มีข้อมูลเพียงพอให้ตอบว่า "ไม่พบข้อมูลในหน้าเว็บนี้"

ให้คุณตัดสินใจรูปแบบคำตอบที่เหมาะสมเองตามคำถาม เช่น:
- ถ้าคำถามเกี่ยวกับ "อยู่หน้าไหน / ลิงก์ / ที่ไหน" ให้ระบุ URL ที่เกี่ยวข้องให้ชัดเจน
- ถ้าคำถามว่า "มีบริการ/มีสินค้าไหม" ให้ระบุว่ามีหรือไม่มี และให้ URL ถ้าพบ
- ถ้าเป็นคำถามทั่วไป ให้สรุปเนื้อหาที่เกี่ยวข้องแบบสั้น กระชับ เข้าใจง่าย

ข้อกำหนด:
- ตอบเฉพาะจากข้อมูลที่มีในบริบทด้านล่างเท่านั้น
- หากสรุป ให้สั้นและชัดเจน
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

def answer_question(question: str):
    # แยกคำคีย์เวิร์ดจากคำถาม
    question_words = question.lower().split()

    # ค้นหาบริบทจากคำถาม
    context = find_best_context(question)
    if not context:
        return "❌ ไม่พบเนื้อหาที่เกี่ยวข้องในฐานข้อมูล"

    # ตรวจสอบและแก้ไขคำที่สะกดผิด
    corrected_question = correct_spelling(question, question_words)

    # ใช้คำที่ได้รับการแก้ไขในการค้นหาอีกครั้ง
    context = find_best_context(corrected_question)
    if not context:
        return "❌ ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล"

    html_content = context.get("HTML", "")[:10000]
    prompt = build_prompt(corrected_question, html_content, context)
    reply = ask_llama(prompt)

    return f"{reply.strip()}\n\n🔗 อ้างอิง: {context.get('URL', 'ไม่พบ URL')}"
