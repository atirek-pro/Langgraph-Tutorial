import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "AGENT_PROJECT"

# --------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# --------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  response = requests.get(url)

  return response.json()

# --------------------------------------------------------------------
# System Prompt
# --------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Use the available tools whenever required.
Always provide the final answer in a clear and concise manner.
"""

# --------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------

agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
    system_prompt=SYSTEM_PROMPT,
)

# --------------------------------------------------------------------
# Invoke
# --------------------------------------------------------------------

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is release date of Avenger's Dooms data?"
            }
        ]
    }
)

print(response)
print(response["messages"][-1].content)