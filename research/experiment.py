import os
import certifi
from dotenv import load_dotenv
import requests
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
WEATHERSTACK_API_KEY = os.environ["WEATHERSTACK_API_KEY"]

@tool
def get_weather_data(city: str) -> str:
    """Get the current weather data for a specified city."""

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

search_tool = TavilySearch(max_results = 3)

# result = search_tool.invoke("What are the current news headlines?")
# print(result)

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0,
    api_key = GOOGLE_API_KEY
)

prompt = "You are a helpful assistant that finds accurate information and answer user questions."

tools = [search_tool, get_weather_data]

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt = prompt
)

response = agent.invoke({
    "messages": [HumanMessage(content="Find the capital of India"
                                      "and then find its current weather")]
})

# Print the final output message from the agent
print(response["messages"][-1].content[0]['text'])