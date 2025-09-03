from vectordb import ChromaDBManager
from ollama_client import OllamaClient


chromadb_path = "../tmp/chromadb"
ollama = OllamaClient()
vectordb = ChromaDBManager(persist_directory=chromadb_path)
collection_name = 'schema'
files_path = ["../tmp/knowledge_base/schema.json", "../tmp/knowledge_base/regras_negocio.pdf"]
prompt_file_path = "../tmp/prompts/data_catalog.txt" # Prompt file for the data catalog

def initialize_knowledge_base(files_path, collection_name: str):
    print("Inicializando base de conhecimento...")
    
    collection, num_documents = vectordb.add_files_to_knowledge_base(files_path, collection_name)
    
    print(f"Base de conhecimento iniciada com sucesso! {num_documents} documentos adicionados.")

    return collection

def query_with_rag( question: str, n_results: int = 3):
        system_message = ollama.load_prompt_file(prompt_file_path)        

        search_results = vectordb.search_in_chromadb(question, collection_name, n_results)
        context = vectordb.get_context_from_search(search_results)
        
        response = ollama.chat_with_context(question, context, system_message)
        
        return response, context
    
def main():
    initialize_knowledge_base(files_path, collection_name)
    
    while True:
        question = input("Digite sua pergunta (ou 'sair' para encerrar): ")
        if question.lower() == 'sair':
            break

        response, context = query_with_rag(question)
        print("Resposta:", response)
        print("Contexto:", context)

if __name__ == "__main__":
    main()