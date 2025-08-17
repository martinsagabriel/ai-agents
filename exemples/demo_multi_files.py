from vectordb import ChromaDBManager
from llm_client import LLMClient
from main import RAGApplication


def demo_multiple_files():
    """Demonstração do sistema RAG com múltiplos tipos de arquivo"""
    
    collection_name = 'demo_multi_files'
    prompt_file_path = "../tmp/prompts/data_catalog.txt"
    
    # Lista de arquivos de diferentes tipos
    file_paths = [
        "../tmp/knowledge_base/schema.json",
        "../tmp/knowledge_base/VeiculosEletricosnoBrasil.pdf",
        "../tmp/knowledge_base/info_veiculos_eletricos.txt"
    ]
    
    rag_app = RAGApplication()
    
    try:
        # Inicializar com múltiplos arquivos
        rag_app.initialize_knowledge_base_from_multiple_files(file_paths, collection_name, prompt_file_path)
        
        print("\n" + "="*50)
        print("DEMO: Sistema RAG com Múltiplos Tipos de Arquivo")
        print("="*50)
        print("\nPerguntas de exemplo que você pode fazer:")
        print("1. 'O que são veículos elétricos?'")
        print("2. 'Quais os desafios dos carros elétricos no Brasil?'")
        print("3. 'Me fale sobre as tabelas do schema'")
        print("4. 'Quais incentivos governamentais existem?'")
        print("\nComandos:")
        print("- 'clear' - Limpar memória")
        print("- 'summary' - Ver resumo da conversa") 
        print("- 'exit' - Sair")
        print("\n" + "="*50)

        while True:
            question = input(f"\nSua pergunta: ")
            
            if question.lower() in ['exit', 'quit', 'sair']:
                print("Até logo!")
                break
            elif question.lower() in ['clear', 'limpar']:
                rag_app.clear_memory()
                continue
            elif question.lower() in ['summary', 'resumo']:
                summary = rag_app.show_conversation_summary()
                print(f"\n{summary}")
                continue

            print("\n📚 Buscando informações...")
            response, context = rag_app.query_with_rag_and_memory(question)
            
            print(f"\n🤖 Resposta:")
            print(f"{response}")
            
            # Mostrar de quais arquivos vieram as informações
            print(f"\n📄 Fontes consultadas:")
            search_results = rag_app.simple_search(question, n_results=3)
            sources = set()
            for metadata in search_results['metadatas'][0]:
                if 'arquivo_origem' in metadata:
                    sources.add(f"- {metadata['arquivo_origem']} ({metadata.get('tipo', 'N/A')})")
            
            for source in sources:
                print(source)

    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    demo_multiple_files()
