import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time
from concurrent.futures import ThreadPoolExecutor

# ฟังก์ชันสำหรับดึงข้อมูล HTML
def fetch_html(url, retries=3, timeout=30):
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, "html.parser")
            
            # เอาเฉพาะเนื้อหาหลักของเว็บ
            html_text = soup.get_text(separator="\n", strip=True)
            return html_text
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching {url}: {e}")
            time.sleep(5)  # ลองใหม่หลังจาก 5 วินาที
    return None

# ฟังก์ชันหลักที่ใช้โหลด CSV และดึงข้อมูล HTML
def main():
    # โหลดไฟล์ CSV
    csv_file = "ข้อมูลลิงก์เว็บ - BIO-INDUSTRIES.csv"
    df = pd.read_csv(csv_file)

    # ตรวจสอบคอลัมน์ที่จำเป็น
    required_cols = ["ID", "URL", "Center", "Header", "NamePage", "Tag"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Column '{col}' not found in the CSV file")

    results = []

    # ดึงข้อมูล HTML ขนานจากหลายๆ URL
    urls = df["URL"].tolist()
    with ThreadPoolExecutor() as executor:
        html_data = list(executor.map(fetch_html, urls))

    # สร้างผลลัพธ์ที่ต้องการ
    for index, row in df.iterrows():
        html_text = html_data[index]  # ดึง HTML ที่ได้จาก ThreadPoolExecutor

        result = {
            "ID": row["ID"],
            "URL": row["URL"],
            "Header": row["Header"],
            "Center": row["Center"],
            "NamePage": row["NamePage"],
            "Tag": row["Tag"],
            "HTML": html_text if html_text else "ไม่สามารถดึงข้อมูลได้"
        }

        results.append(result)

    # บันทึกผลลัพธ์เป็นไฟล์ JSON
    output_file = "output_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ Data has been saved to output_data.json")

# เรียกใช้ฟังก์ชันหลัก
if __name__ == "__main__":
    main()
