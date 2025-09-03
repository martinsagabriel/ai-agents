from dotenv import load_dotenv
import os

from groq import Groq

load_dotenv('../.env')
MODEL = 'openai/gpt-oss-120b'

RESET = '\033[0m'
BOLD = '\033[1m'

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'

client = Groq(
  api_key=os.getenv('GROQ_API_KEY')
)

def chat(prompt: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
        }
    ]
)
    return completion.choices[0].message.content

session = True

while session == True:
    user_question = input(f"\n{BOLD}{YELLOW}>> {RESET}")

    if user_question.lower() in ['exit', 'quit', 'sair', 'bye']:
        print(f"{BOLD}{RED}Saindo da aplicação...{RESET}")
        break

    response = chat(user_question)
    print(f"\n{BOLD}{GREEN}Resposta:{RESET}")
    print(f"{GREEN}{response}{RESET}")