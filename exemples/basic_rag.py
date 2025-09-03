from dotenv import load_dotenv
import os
import json
import chromadb
from chromadb.config import Settings

from groq import Groq

load_dotenv('../.env')

MODEL = 'openai/gpt-oss-120b'

client = Groq(
    api_key=os.getenv('GROQ_API_KEY')
)

chroma_client = chromadb.PersistentClient(path="../tmp/chromadb") # Configurar ChromaDB local

def load_json_data(json_path: str):
    """Carrega dados de um arquivo JSON"""
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_prompt_file(file_path: str):
    """Carrega um arquivo de prompt"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def create_or_get_collection(collection_name: str):
    """Cria ou obtém uma coleção no ChromaDB"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        collection = chroma_client.create_collection(name=collection_name)
    return collection

def add_data_to_chromadb(data, collection):
    """Adiciona dados JSON ao ChromaDB"""
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

def search_in_chromadb(query: str, collection, n_results: int = 3):
    """Busca por informações relevantes no ChromaDB"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

def chat_with_context(prompt: str, context_data: str = "", system_message: str = "") -> str:
    """Chat com contexto das informações do banco de dados"""
    
    system_message = system_message.format(context_data=context_data)

    print("system_message:", system_message)

    print("prompt:", prompt)

    # completion = client.chat.completions.create(
    #     model=MODEL,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": system_message
    #         },
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ]
    # )
    # return completion.choices[0].message.content

def initialize_knowledge_base(file_path: str, file_type = ''):
    """Inicializa a base de conhecimento carregando dados do JSON"""
    data = load_json_data(file_path)
    
    collection = create_or_get_collection(collection_name)
    num_documents = add_data_to_chromadb(data, collection)
    
    # print(f"Base de conhecimento inicializada com {num_documents} documentos.")
    return collection

def query_with_rag(question: str, collection):
    """Faz uma pergunta usando RAG (Retrieval-Augmented Generation)"""
    search_results = search_in_chromadb(question, collection)
    context = ""
    
    if search_results['documents'][0]:
        context = "\n\n".join(search_results['documents'][0])
    response = chat_with_context(question, context)
    
    return response, context

if __name__ == "__main__":
    collection_name = 'data_catalog'
    json_path = "../tmp/knowledge_base/schema.json"
    system_message = load_prompt_file("../tmp/prompts/data_catalog.txt")
    
    # Inicializar base de conhecimento
    collection = initialize_knowledge_base(json_path)
    
    # Fazer uma busca simples
    question = "Como descubro a quantidade de produtos comprados em um pedido?" # input("Digite sua pergunta: ")
    # search_results = search_in_chromadb(question, collection)
    
    response, context = query_with_rag(question, collection)
    print(f"Resposta: {response}")