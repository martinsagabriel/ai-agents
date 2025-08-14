from agno.agent import Agent
from agno.models.groq import Groq

from agno.playground import Playground, serve_playground_app

from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.chroma import ChromaDb

vector_db = ChromaDb(
    collection="recipes",
    path="../tmp/chromadb",
    persistent_cliente=True
)


knowledge_base = PDFKnowledgeBase(
    path="../tmp/knowledge_base/VeiculosEletricosnoBrasil.pdf",
    vector_db=vector_db,
    reader=PDFReader(chunk=True)
)
 
agent = Agent(
    model=Groq(id="openai/gpt-oss-20b"),
    knowledge=knowledge_base,
    search_knowledge=True,
)

app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("pdf_agent:app", reload=True)