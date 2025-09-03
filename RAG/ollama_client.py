import requests

class OllamaClient:
    def __init__(self, url: str = ''):
        self.url = "http://localhost:11434"
        
    def list_models(self):
        """
        Lista os modelos disponíveis no Ollama
        
        Returns:
            Lista de modelos disponíveis
        """
        try:
            response = requests.get(f"{self.url}/api/tags")
            if response.status_code == 200:
                result = response.json()
                print(f"Modelos disponíveis: {', '.join([model['name'] for model in result.get('models', [])])}")
                return [model['name'] for model in result.get('models', [])]
            else:
                return []
        except:
            return []
        
    def select_model(self):
        models = self.list_models()
        if not models:
            print("Nenhum modelo disponível.")
            return None

        print("Selecione um modelo:")
        for i, model in enumerate(models):
            print(f"{i + 1}. {model}")

        choice = input("Digite o número do modelo desejado: ")
        try:
            model_index = int(choice) - 1
            if 0 <= model_index < len(models):
                return models[model_index]
            else:
                print("Modelo inválido.")
                return None
        except ValueError:
            print("Entrada inválida.")
            return None 

    def load_prompt_file(self, file_path):
        """
        Carrega o conteúdo de um arquivo de prompt.
        """
        if file_path == "" or file_path == None:
                file_path = "../tmp/prompts/base_prompt.txt"
                
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def chat_with_ollama(self, prompt, model):
        messages = []
        """
        Envia uma mensagem para o modelo Ollama e retorna a resposta.
        """
        messages.append({
            "role": "user",
            "content": prompt
        })
            
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(
            f"{self.url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        ).json()
        
        return response['message']['content']

    def chat_with_context(self, prompt, context_data = "", system_message = "", model=""):
        formatted_system_message = system_message.format(context_data=context_data)
        messages = []

        messages.append({
            "role": "user",
            "content": formatted_system_message
        })

        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }

        response = requests.post(
            f"{self.url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        ).json()

        return response['message']['content']
