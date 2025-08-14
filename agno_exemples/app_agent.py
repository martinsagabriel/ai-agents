from agno.agent import Agent
from agno.models.groq import Groq
from agno.playground import Playground, serve_playground_app

from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=Groq(id="openai/gpt-oss-20b"),
    tools=[YFinanceTools(stock_price=True)],
    instructions="use tabelas para mostrar a informação final, nao incluir texto adicional.",
    markdown=True,
)

app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("app_agent:app", reload=True)