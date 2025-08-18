import streamlit as st
from qa_engine import answer_question

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="🔍 TISTR AI Search", layout="centered")
st.title("🔍 ระบบ AI ถาม-ตอบจากเว็บไซต์ TISTR")
st.markdown("ยินดีต้อนรับสู่ระบบ AI ถาม-ตอบจากเว็บไซต์ TISTR! คุณสามารถถามคำถามเกี่ยวกับบริการ ผลิตภัณฑ์ หรือเนื้อหาอื่นๆ ที่เกี่ยวข้องกับ TISTR ได้ที่นี่")
# โหลดคำถามยอดนิยมจาก session_state หรือกำหนดค่าเริ่มต้น
if "popular_questions" not in st.session_state:
    st.session_state.popular_questions = [
        "TISTR มีบริการอะไรบ้าง?",
        "ผลิตภัณฑ์ที่ TISTR มีคืออะไร?",
        "ติดต่อ TISTR ได้ที่ไหน?",
        "TISTR มีศูนย์บริการที่ไหนบ้าง?",
        "TISTR ทำงานเกี่ยวกับอะไร?"
    ]

# แสดงคำถามยอดนิยม
st.markdown("### ⭐ คำถามยอดนิยม")
cols = st.columns(2)
for i, q in enumerate(st.session_state.popular_questions):
    if cols[i % 2].button(f"💬 {q}"):
        st.session_state.selected_question = q

# ฟิลด์กรอกคำถาม
question = st.text_input("❓ ใส่คำถามเกี่ยวกับบริการ ผลิตภัณฑ์ หรือเนื้อหาอื่นๆ", value=st.session_state.get("selected_question", ""))

# ส่งคำถามและรับคำตอบ
if st.button("📤 ถามเลย") and question:
    with st.spinner("🧠 กำลังประมวลผล..."):
        answer = answer_question(question)
        st.markdown("### 📄 คำตอบ")
        st.markdown(answer)
    st.session_state.selected_question = ""  # รีเซ็ตคำถามหลังการตอบ