# ollama_utils.py
import ollama

def ask_llama(prompt: str, model: str = "gemma2") -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "คุณคือผู้ช่วยภาษาไทยสำหรับตอบคำถามจากเนื้อหาเว็บไซต์"},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content'].strip()