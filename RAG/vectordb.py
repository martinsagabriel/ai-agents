import json
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Any
import os
from file_process import FileProcessor


class ChromaDBManager:
    
    def __init__(self, persist_directory: str = "../tmp/chromadb"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.file_processor = FileProcessor(file_path="")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Tenta quebrar no final de uma frase
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size * 0.5:  # Só quebra se for pelo menos na metade
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap if end < len(text) else end
            
        return chunks
    
    def create_or_get_collection(self, collection_name: str):
        try:
            collection = self.client.get_collection(name=collection_name)
        except:
            collection = self.client.create_collection(name=collection_name)
        return collection
    
    def add_data_to_chromadb(self, data: Any, collection, data_type: str = 'json', file_name: str = '') -> int:
        documents = []
        metadatas = []
        ids = []
        
        if data_type == 'json' and "tabelas" in data:
            # Processamento de dados JSON (schema de banco)
            for i, tabela in enumerate(data["tabelas"]):
                doc_text = f"Tabela: {tabela['nome']}\nDescrição: {tabela['descricao']}\nCampos:\n"
                
                for campo in tabela['campos']:
                    doc_text += f"- {campo['nome']} ({campo['tipo']}): {campo['descricao']}\n"
                
                documents.append(doc_text)
                metadatas.append({
                    "nome_tabela": tabela['nome'],
                    "tipo": "schema_tabela",
                    "descricao": tabela['descricao'],
                    "arquivo_origem": file_name
                })
                ids.append(f"tabela_{i}")
                
        elif data_type in ['pdf', 'doc', 'txt']:
            # Processamento de texto (PDF, DOC, TXT)
            text_chunks = self.chunk_text(data)
            
            for i, chunk in enumerate(text_chunks):
                documents.append(chunk)
                metadatas.append({
                    "tipo": f"documento_{data_type}",
                    "chunk_id": i,
                    "arquivo_origem": file_name,
                    "total_chunks": len(text_chunks)
                })
                ids.append(f"{file_name}_{data_type}_chunk_{i}")
        
        elif data_type == 'json':
            # Processamento genérico de JSON
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            text_chunks = self.chunk_text(json_str)
            
            for i, chunk in enumerate(text_chunks):
                documents.append(chunk)
                metadatas.append({
                    "tipo": "documento_json",
                    "chunk_id": i,
                    "arquivo_origem": file_name,
                    "total_chunks": len(text_chunks)
                })
                ids.append(f"{file_name}_json_chunk_{i}")
        
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return len(documents)
    
    def search_in_chromadb(self, query: str, collection, n_results: int = 3) -> Dict[str, Any]:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    def initialize_knowledge_base(self, file_path: str, collection_name: str):
        file_name = os.path.basename(file_path)
        data_type, data = self.file_processor.detect_file_type_and_load(file_path)
        collection = self.create_or_get_collection(collection_name)
        num_documents = self.add_data_to_chromadb(data, collection, data_type, file_name)
        return collection, num_documents
    
    def add_files_to_knowledge_base(self, file_paths: List[str], collection_name: str):
        collection = self.create_or_get_collection(collection_name)
        total_documents = 0
        
        for file_path in file_paths:
            try:
                file_name = os.path.basename(file_path)
                data_type, data = self.file_processor.detect_file_type_and_load(file_path)
                num_documents = self.add_data_to_chromadb(data, collection, data_type, file_name)
                total_documents += num_documents
                print(f"Adicionado: {file_name} ({data_type}) - {num_documents} documentos")
            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")
        
        return collection, total_documents
    
    def get_context_from_search(self, search_results: Dict[str, Any]) -> str:
        context = ""
        if search_results['documents'][0]:
            context = "\n\n".join(search_results['documents'][0])
        return context
