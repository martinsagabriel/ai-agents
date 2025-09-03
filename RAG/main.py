from vectordb import ChromaDBManager
from llm_client import LLMClient
from typing import Tuple, List

RESET = '\033[0m'
BOLD = '\033[1m'

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'

class RAGApplication:
    
    def __init__(self, 
                 chromadb_path: str = "../tmp/chromadb",
                 model: str = 'openai/gpt-oss-120b'):
        self.vectordb = ChromaDBManager(persist_directory=chromadb_path)
        self.llm_client = LLMClient(model=model)
        self.collection = None
        self.system_message = ""
    
    def initialize_knowledge_base(self, files_path: List[str], collection_name: str, prompt_file_path: str):
        print("Inicializando base de conhecimento...")
        
        self.system_message = self.llm_client.load_prompt_file(prompt_file_path)        
        self.collection, num_documents = self.vectordb.add_files_to_knowledge_base(files_path, collection_name)
        
        print(f"Base de conhecimento iniciada com sucesso! {num_documents} documentos adicionados.")
    
        return self.collection
    
    def query_with_rag(self, question: str, n_results: int = 3) -> Tuple[str, str]:
        if not self.collection:
            raise ValueError("Base de conhecimento não foi inicializada. Chame initialize_knowledge_base() primeiro.")
        
        search_results = self.vectordb.search_in_chromadb(question, self.collection, n_results)
        context = self.vectordb.get_context_from_search(search_results)
        
        response = self.llm_client.chat_with_context(question, context, self.system_message)
        
        return response, context
    
    def simple_search(self, query: str, n_results: int = 3) -> dict:
        if not self.collection:
            raise ValueError("Base de conhecimento não foi inicializada.")
        
        return self.vectordb.search_in_chromadb(query, self.collection, n_results)
    
    def chat_without_context(self, prompt: str) -> str:
        return self.llm_client.simple_chat(prompt, self.system_message)


def main():
    collection_name = 'schema'
    files_path = ["../tmp/knowledge_base/schema.json", "../tmp/knowledge_base/regras_negocio.pdf"]
    prompt_file_path = "../tmp/prompts/data_catalog.txt" # Prompt file for the data catalog

    rag_app = RAGApplication()
    
    try:
        rag_app.initialize_knowledge_base(files_path, collection_name, prompt_file_path)

        while True:
            
            question = input(f"{BOLD}{YELLOW}{'=' * 20} Pergunta {'=' * 20}{RESET}\n")
            
            if question.lower() in ['exit', 'quit', 'sair', 'bye']:
                print(f"{BOLD}{RED}Saindo da aplicação...{RESET}")
                break

            print(f"\n{BOLD}{GREEN}{'=' * 20} Resposta com RAG {'=' * 20}{RESET}")
            response, context = rag_app.query_with_rag(question)
            # print(f"Contexto encontrado:\n{context}")
            print(f"{response}\n")
        
        # # Exemplo de busca simples no vector database
        # print("\n=== Busca simples no vector database ===")
        # search_results = rag_app.simple_search(question)
        # print("Documentos encontrados:")  
        # for i, doc in enumerate(search_results['documents'][0]):
        #     print(f"{i+1}. {doc[:100]}...")
    
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()
