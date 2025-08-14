from agno.agent import Agent
from agno.models.groq import Groq
from agno.playground import Playground, serve_playground_app

from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.chroma import ChromaDb

tmp_dir = '../tmp'

def load_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

prompt = load_file(f"{tmp_dir}/prompts/data_catalog.txt")

knowledge_base = JSONKnowledgeBase(
    path=f"{tmp_dir}/knowledge_base/schema.json",
    vector_db=ChromaDb(collection="schemas", path=f"{tmp_dir}/chromadb")
)

agent = Agent(
    model=Groq(id="openai/gpt-oss-20b"),
    knowledge=knowledge_base,
    search_knowledge=True,
    instructions=prompt,
)

agent.print_response(f"Quais informações tenho sobre formas de pagamento?", stream=True)