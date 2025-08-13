from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=Groq(id="openai/gpt-oss-20b"),
    tools=[YFinanceTools(stock_price=True)],
    instructions="Use tables to display data. Don't include any other text.",
    markdown=True,
)

agent.print_response("What is the stock price of Apple?", stream=True)