import json
import os
import PyPDF2
from docx import Document
from typing import Dict, Any

class FileProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_json_data(self, json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def load_pdf_data(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Erro ao ler PDF {pdf_path}: {e}")
        return text
    
    def load_doc_data(self, doc_path: str) -> str:
        text = ""
        try:
            doc = Document(doc_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Erro ao ler DOC {doc_path}: {e}")
        return text
    
    def load_txt_data(self, txt_path: str) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Erro ao ler TXT {txt_path}: {e}")
            return ""
    
    def detect_file_type_and_load(self, file_path: str) -> tuple[str, Any]:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.json':
            return 'json', self.load_json_data(file_path)
        elif file_extension == '.pdf':
            return 'pdf', self.load_pdf_data(file_path)
        elif file_extension in ['.doc', '.docx']:
            return 'doc', self.load_doc_data(file_path)
        elif file_extension == '.txt':
            return 'txt', self.load_txt_data(file_path)
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {file_extension}")