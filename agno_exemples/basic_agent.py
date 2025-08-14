from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=Groq(id="openai/gpt-oss-20b"),
    tools=[YFinanceTools(stock_price=True)],
    instructions="Use tables to display data. Don't include any other text.",
    markdown=True,
)

stock = input("Qual a contacao que deseja pesquisar? >> ")

agent.print_response(f"What is the stock price of {stock}?", stream=True)