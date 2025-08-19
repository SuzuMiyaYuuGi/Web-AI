import streamlit as st
from qa_engine import answer_question

# ---------------- NEW: imports & helpers ----------------
import pandas as pd
import os

# ตั้งค่าหน้าเว็บ — กว้างเต็มจอ
st.set_page_config(page_title="🔍 TISTR AI Search", layout="wide")

# ฉีด CSS ให้กว้างเต็มหน้า + ขยายป๊อปอัป
st.markdown("""
<style>
/* ทำให้คอนเทนต์หลักกว้างเต็มและมี padding อ่านง่าย */
.main .block-container {
  max-width: 1400px;
  padding-left: 2rem;
  padding-right: 2rem;
}

/* ปรับความกว้าง/สูงของ dialog (ป๊อปอัป) */
div[role="dialog"] {
  width: 92vw !important;
  max-width: 1400px !important;
}

/* ทำให้เนื้อหาในป๊อปอัปเลื่อนภายในได้ และสูงตามจอ */
div[role="dialog"] > div {
  max-height: 82vh;
  overflow: auto;
  padding-bottom: 0.5rem;
}

/* เพิ่มพื้นที่ให้ DataFrame ในป๊อปอัป */
div[role="dialog"] .stDataFrame {
  height: 70vh !important;
}

/* ปุ่มให้ดูชัด */
button[kind="primary"] {
  transform: translateZ(0);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_products(csv_path="Product List.csv"):
    """โหลดสินค้า + ทำชื่อคอลัมน์ให้มาตรฐาน: ID / ชื่อสินค้า / ศูนย์ / link"""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["ID", "ชื่อสินค้า", "ศูนย์", "link"])

    # รองรับหลาย encoding
    encodings = [None, "utf-8-sig", "cp874", "latin-1"]
    last_err = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc) if enc else pd.read_csv(csv_path)
            break
        except Exception as e:
            last_err = e
    if df is None:
        st.error(f"ไม่สามารถอ่านไฟล์สินค้าได้: {last_err}")
        return pd.DataFrame(columns=["ID", "ชื่อสินค้า", "ศูนย์", "link"])

    # ทำให้ชื่อคอลัมน์ยืดหยุ่นขึ้น
    rename_map = {}
    cols_norm = {c.lower().strip(): c for c in df.columns}

    if "id" in cols_norm: rename_map[cols_norm["id"]] = "ID"
    for key in ["name", "product", "product name", "ชื่อสินค้า", "รายชื่อสินค้า", "รายการสินค้า"]:
        if key in cols_norm: rename_map[cols_norm[key]] = "ชื่อสินค้า"; break
    for key in ["center", "ศูนย์", "ศูนย์งาน", "หน่วยงาน"]:
        if key in cols_norm: rename_map[cols_norm[key]] = "ศูนย์"; break
    # ✅ รองรับทั้ง "ลิงก์" และ "ลิ้งค์" + url
    for key in ["link", "url", "ลิงก์", "ลิ้งค์"]:
        if key in cols_norm: rename_map[cols_norm[key]] = "link"; break

    df = df.rename(columns=rename_map)

    # ให้มีคอลัมน์มาตรฐานครบ
    for col in ["ID", "ชื่อสินค้า", "ศูนย์", "link"]:
        if col not in df.columns:
            df[col] = ""

    # ตัดช่องว่างหัว–ท้าย ป้องกันกรองไม่ติด
    for col in ["ID", "ชื่อสินค้า", "ศูนย์", "link"]:
        df[col] = df[col].astype(str).fillna("").str.strip()

    # จัดลำดับคอลัมน์
    df = df[["ID", "ชื่อสินค้า", "ศูนย์", "link"]].copy()
    return df


def _is_valid_url(u: str) -> bool:
    u = str(u or "").strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def _filter_products(df, keyword="", center="ทั้งหมด"):
    if df.empty:
        return df

    filtered = df.copy()

    # ใช้ regex=False เพื่อแมตช์ชื่อศูนย์ตรง ๆ (กันอักขระพิเศษทำพัง)
    if center and center != "ทั้งหมด":
        filtered = filtered[
            filtered["ศูนย์"].str.contains(center, case=False, na=False, regex=False)
        ]

    if keyword:
        kw = keyword.strip().lower()
        mask = (
            filtered["ID"].str.lower().str.contains(kw, na=False) |
            filtered["ชื่อสินค้า"].str.lower().str.contains(kw, na=False) |
            filtered["ศูนย์"].str.lower().str.contains(kw, na=False) |
            filtered["link"].str.lower().str.contains(kw, na=False)
        )
        filtered = filtered[mask]

    return filtered.reset_index(drop=True)


def _products_ui(df):
    # ตัวกรอง
    cols = st.columns([2, 1])
    keyword = cols[0].text_input("🔎 ค้นหา", placeholder="พิมพ์คำค้น เช่น ชื่อสินค้า / ID / ศูนย์ ...", key="products_kw")
    centers = ["ทั้งหมด"] + sorted([c for c in df["ศูนย์"].dropna().astype(str).unique() if c.strip()])
    center = cols[1].selectbox("ศูนย์", centers, index=0, key="products_center")

    filtered = _filter_products(df, keyword, center)

    # ✅ ปรับคอลัมน์ link:
    # - ถ้าเป็น URL: คงค่า URL ไว้ → แสดงเป็นลิงก์กดได้
    # - ถ้าไม่มี / ว่าง / NaN: ให้เป็นค่าว่าง
    df_show = filtered.copy()
    df_show["link"] = df_show["link"].apply(lambda u: (u if _is_valid_url(u) else ""))

    # แสดงตาราง
    st.dataframe(
        df_show[["ID", "ชื่อสินค้า", "ศูนย์", "link"]],
        use_container_width=True,
        height=560,
        column_config={
            "link": st.column_config.LinkColumn("link", display_text="ไปยังหน้าเว็บ"),
        }
    )

    # ปุ่มดาวน์โหลดเฉพาะผลกรอง
    csv = df_show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ ดาวน์โหลดผลลัพธ์ (CSV)", data=csv, file_name="products_filtered.csv", mime="text/csv")


# ---------------- Header / Intro ----------------
st.title("🔍 ระบบ AI ถาม-ตอบจากเว็บไซต์ TISTR")
st.markdown("ยินดีต้อนรับสู่ระบบ AI ถาม-ตอบจากเว็บไซต์ TISTR! คุณสามารถถามคำถามเกี่ยวกับบริการ ผลิตภัณฑ์ หรือเนื้อหาอื่นๆ ที่เกี่ยวข้องกับ TISTR ได้ที่นี่")

# โหลดคำถามยอดนิยมจาก session_state หรือกำหนดค่าเริ่มต้น
if "popular_questions" not in st.session_state:
    st.session_state.popular_questions = []

# ---------------- NEW: โหลดสินค้าหนึ่งครั้ง ----------------
if "products_df" not in st.session_state:
    st.session_state.products_df = _load_products()

# ---------------- NEW: ปุ่มเปิดป๊อปอัปสินค้า ----------------
st.markdown("### 🧾 รายการสินค้า")
open_btn = st.button("📋 รายชื่อสินค้า/บริการ")

# พยายามใช้ st.dialog (ถ้ามี) ให้เป็นป๊อปอัป; ถ้าไม่มีใช้ fallback
if hasattr(st, "dialog"):
    @st.dialog("🧾 รายการสินค้า (เลื่อนและค้นหาได้)")
    def products_modal():
        _products_ui(st.session_state.products_df)

    if open_btn:
        products_modal()
else:
    # Fallback: เรนเดอร์ในส่วนขยายบนหน้า (ไม่ป๊อปอัป แต่เต็มความกว้าง)
    if open_btn:
        with st.expander("🧾 รายการสินค้า (Fallback โหมด)", expanded=True):
            _products_ui(st.session_state.products_df)

# ---------------- เดิม: ฟิลด์ถาม-ตอบ ----------------
question = st.text_input(
    "❓ ใส่คำถามเกี่ยวกับบริการ ผลิตภัณฑ์ หรือเนื้อหาอื่นๆ",
    value=st.session_state.get("selected_question", "")
)

if st.button("📤 ถามเลย") and question:
    with st.spinner("🧠 กำลังประมวลผล..."):
        answer = answer_question(question)
        st.markdown("### 📄 คำตอบ")
        st.markdown(answer)
    st.session_state.selected_question = ""  # รีเซ็ตคำถามหลังการตอบ
