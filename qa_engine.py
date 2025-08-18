import json
from ollama_utils import ask_llama
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data from JSON
with open("output_data.json", encoding="utf-8") as f:
    docs = json.load(f)

# Improved Token Matching: Lemmatization or stemming (if applicable for Thai language)
def tokenize_and_clean(question):
    # Example: could implement stemming or lemmatization here (e.g., using pythainlp's stem function)
    return word_tokenize(question.lower(), engine='newmm')

# Use TF-IDF for more context-aware document retrieval
def find_best_context(question: str):
    question_words = tokenize_and_clean(question)
    
    # Build the document corpus including HTML and metadata text for TF-IDF
    corpus = [entry.get("HTML", "") + " " + entry.get("Header", "") + " " + entry.get("Tag", "") for entry in docs]
    
    # Convert to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Vectorize the question
    question_vec = vectorizer.transform([" ".join(question_words)])

    # Compute cosine similarity between question and each document
    similarities = tfidf_matrix * question_vec.T
    scores = similarities.toarray().flatten()

    # Get the best matching document based on similarity score
    best_match_idx = scores.argmax()
    if scores[best_match_idx] == 0:
        return None

    return docs[best_match_idx]

# Combine metadata into the context to enhance response generation
def build_prompt(question: str, context_html: str, metadata: dict):
    base = """
กรุณาตอบเป็นภาษาไทยเท่านั้น โดยอ้างอิงจากเนื้อหาของหน้าเว็บไซต์ด้านล่าง
นำข้อมูลทั้งหมดมาตอบเลยและสรุปให้กระชับและเข้าใจง่าย

ข้อกำหนด:
- ตอบเฉพาะจากข้อมูลที่มีในบริบทด้านล่างเท่านั้น
- หากสรุป ให้สั้นและชัดเจนแต่รายละเอียดครบถ้วน
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

# Function to answer the question based on the context
def answer_question(question: str):
    context = find_best_context(question)
    if not context:
        return "❌ ไม่พบเนื้อหาที่เกี่ยวข้องในฐานข้อมูล"

    html_content = context.get("HTML", "")[:10000]  # Limit content to the first 10,000 characters
    prompt = build_prompt(question, html_content, context)
    reply = ask_llama(prompt)

    return f"{reply.strip()}\n\n🔗 อ้างอิง: {context.get('URL', 'ไม่พบ URL')}"
