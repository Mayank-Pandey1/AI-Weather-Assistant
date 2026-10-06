
import os

import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage


# -------------------- Configuration --------------------

st.set_page_config(
    page_title="AI Weather Assistant",
    page_icon="🌦️",
    layout="centered",
)

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()


# -------------------- Page UI --------------------

st.title("🌦️ AI Weather Assistant")
st.caption(
    "Ask questions, search the web, and check current weather "
    "using a Gemini-powered AI agent."
)

with st.sidebar:
    st.header("About this assistant")
    st.write(
        "An AI agent powered by Gemini that can search the web "
        "and retrieve current weather information."
    )

    st.subheader("Available tools")
    st.markdown(
        "- 🔎 **Tavily Search:** Web search\n"
        "- 🌡️ **Weatherstack:** Current weather"
    )

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# -------------------- Weather Tool --------------------

@tool
def get_weather_data(city: str) -> str:
    """Get the current weather data for a specified city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "Weatherstack API key is not configured."

    url = "http://api.weatherstack.com/current"

    try:
        response = requests.get(
            url,
            params={
                "access_key": api_key,
                "query": city,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if "current" not in data:
            return (
                f"Could not fetch weather for {city}. "
                f"API response: {data.get('error', {}).get('info', 'Unknown error')}"
            )

        current = data["current"]
        location = data.get("location", {})

        return (
            f"City: {location.get('name', city)}\n"
            f"Country: {location.get('country', 'Unknown')}\n"
            f"Temperature: {current['temperature']}°C\n"
            f"Weather: {current['weather_descriptions'][0]}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind speed: {current.get('wind_speed', 'Unknown')} km/h"
        )

    except requests.RequestException as exc:
        return f"Weather service request failed: {exc}"
    except (ValueError, KeyError, IndexError) as exc:
        return f"Could not process weather data: {exc}"


# -------------------- Create Agent --------------------

@st.cache_resource
def build_agent():
    required_keys = [
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
        "WEATHERSTACK_API_KEY",
    ]

    missing_keys = [
        key for key in required_keys if not os.getenv(key)
    ]

    if missing_keys:
        raise ValueError(
            "Missing API keys in .env: " + ", ".join(missing_keys)
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    search_tool = TavilySearch(max_results=3)

    tools = [search_tool, get_weather_data]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant that finds accurate information "
            "and answers user questions. Use web search for information "
            "that requires research or current facts. Use the weather tool "
            "when the user asks about current weather. When asked to find "
            "a location and its weather, identify the location first, then "
            "retrieve its weather. Clearly explain the results."
        ),
    )


# -------------------- Chat History --------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------- Chat Input --------------------

user_input = st.chat_input(
    "Ask about a place, the weather, or anything else..."
)

if user_input:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and gathering information..."):
            try:
                agent = build_agent()

                response = agent.invoke({
                    "messages": [HumanMessage(content=user_input)]
                })

                # Extract the final response safely
                content = response["messages"][-1].content

                if isinstance(content, list):
                    answer = "\n\n".join(
                        block["text"]
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        and block.get("text")
                    )
                elif isinstance(content, str):
                    answer = content
                else:
                    answer = str(content)

                if not answer.strip():
                    answer = "The agent returned an empty response."

                st.markdown(answer)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")