import streamlit as st
from qa_engine import answer_question

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="🔍 TISTR AI Search", layout="centered")
st.title("🔍 ระบบ AI ถาม-ตอบจากเว็บไซต์ TISTR")
st.caption("✅ ใช้ LLaMA3 ผ่าน Ollama (รันในเครื่อง) พร้อมลิงก์อ้างอิง")

# โหลดคำถามยอดนิยมจาก session_state หรือกำหนดค่าเริ่มต้น
if "popular_questions" not in st.session_state:
    st.session_state.popular_questions = [
        "จะติดต่อ InnoAG ได้อย่างไร",
        "จะติดต่อ InnoFOOD ได้อย่างไร",
        "จะติดต่อ InnoHerb ได้อย่างไร",
        "จะติดต่อ InnoEn ได้อย่างไร",
        "จะติดต่อ InnoRobot ได้อย่างไร",
        "จะติดต่อ InnoMat ได้อย่างไร",
        "จะติดต่อ BRC ได้อย่างไร",
        "จะติดต่อ RTTC ได้อย่างไร",
        "จะติดต่อ MPAD ได้อย่างไร",
    ]

# เพิ่มคำถามยอดนิยมใหม่
with st.expander("➕ เพิ่มคำถามยอดนิยม"):
    new_q = st.text_input("📌 เพิ่มคำถามใหม่ที่นี่")
    if st.button("➕ บันทึกคำถาม"):
        if new_q.strip() != "" and new_q not in st.session_state.popular_questions:
            st.session_state.popular_questions.append(new_q)
            st.success("✅ เพิ่มคำถามเรียบร้อยแล้ว!")

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
