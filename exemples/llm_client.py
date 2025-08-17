from dotenv import load_dotenv
import os
from typing import Optional
from groq import Groq

load_dotenv('../.env')

class LLMClient:
    def __init__(self, model: str = ''):
        self.model = model or 'meta-llama/llama-guard-4-12b'
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    def load_prompt_file(self, file_path: Optional[str] = None) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chat_with_context(self, prompt: str, context_data: str = "", system_message: str = "") -> str:
        formatted_system_message = system_message.format(context_data=context_data)
        
        try:
            # print("System message:", formatted_system_message[:200] + "..." if len(formatted_system_message) > 200 else formatted_system_message)
            # print("User prompt:", prompt)
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": formatted_system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Erro ao chamar a API: {e}")
            return f"Erro ao processar a solicitação: {str(e)}"
    
    def simple_chat(self, prompt: str, system_message: Optional[str] = None) -> str:
        messages = []
        
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Erro ao chamar a API: {e}")
            return f"Erro ao processar a solicitação: {str(e)}"
