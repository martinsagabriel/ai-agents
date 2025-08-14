import json
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Any, Optional


class ChromaDBManager:
    
    def __init__(self, persist_directory: str = "../tmp/chromadb"):
        self.client = chromadb.PersistentClient(path=persist_directory)
    
    def load_json_data(self, json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def create_or_get_collection(self, collection_name: str):
        try:
            collection = self.client.get_collection(name=collection_name)
        except:
            collection = self.client.create_collection(name=collection_name)
        return collection
    
    def add_data_to_chromadb(self, data: Dict[str, Any], collection) -> int:
        documents = []
        metadatas = []
        ids = []
        
        if "tabelas" in data:
            for i, tabela in enumerate(data["tabelas"]):
                doc_text = f"Tabela: {tabela['nome']}\nDescrição: {tabela['descricao']}\nCampos:\n"
                
                for campo in tabela['campos']:
                    doc_text += f"- {campo['nome']} ({campo['tipo']}): {campo['descricao']}\n"
                
                documents.append(doc_text)
                metadatas.append({
                    "nome_tabela": tabela['nome'],
                    "tipo": "schema_tabela",
                    "descricao": tabela['descricao']
                })
                ids.append(f"tabela_{i}")
        
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
        data = self.load_json_data(file_path)
        collection = self.create_or_get_collection(collection_name)
        num_documents = self.add_data_to_chromadb(data, collection)
        return collection, num_documents
    
    def get_context_from_search(self, search_results: Dict[str, Any]) -> str:
        context = ""
        if search_results['documents'][0]:
            context = "\n\n".join(search_results['documents'][0])
        return context
