import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

from typing import  TypedDict, Annotated
import operator
import uuid

import psycopg # Python talk to a PostgreSQL database
from psycopg.rows import dict_row ## changes the row from the db into dict.

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
# from tools.flight_tool import search_flights
from mcp_client import fetch_flight_reference_data, tavily_mcp_search, weather_mcp_multi_forecast, weather_mcp_forecast
import asyncio

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. PLease add your Render PostgreSQL External DB URL to .env"
        )
    if "sslmode" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. add it to the env.")


# =========================
# LLM
# =========================

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key = GROQ_API_KEY
)

# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add] # messages keep on appending by the reducer(operator.add())
    user_query: str
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str
    llm_calls: int


# =========================
# Flight Agent
# =========================


 # def flight_agent(state: TravelState):
 #     query = state["user_query"]
  #  flight_data = search_flights(query)
#
 #   return {
  #      "flight_results": flight_data,
   #     "messages": [
    #        AIMessage(content = "Flight results fetched")
     #   ],
      #  "llm_calls": state.get("llm_calls", 0)+1
   # }

## Flight tool initaialization from mcp:
FLIGHT_AGENT_PROMPT = """
                    You are a travel flight expert.

                    User Query:
                    {query}
                    
                    Airport Information:
                    {airport_data}
                    
                    Airline Information:
                    {airline_data}
                    
                    Generate:
                    
                    1. Likely departure airport
                    2. Likely arrival airport
                    3. Typical flight duration
                    5. Estimated airfare
                    6. Peak season pricing warning
                    7. Booking advice
                    
                    Return concise travel guidance"""

# Flight Agent
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]
    flight_data = None  # default so it's always defined

    try:
        airports, airlines = asyncio.run(fetch_flight_reference_data())

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=airports[:3000],
            airline_data=airlines[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are a travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:
        print(f"Flight information unavailable: {e}")

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = asyncio.run(tavily_mcp_search(query))

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# ==========================
# Weather Agent
# ==========================


import json

WEATHER_CITY_EXTRACTION_PROMPT = """
Analyze this travel query and determine the destination(s).

Query: {query}

Rules:
- If the query names a specific city, return just that one city.
- If the query names a country or region (not a specific city), return 3-5 major cities
  a typical first-time tourist itinerary for that country would include.
- Use well-known city names only (no ambiguous or made-up names).

Respond with ONLY valid JSON, no other text, in this exact shape:
{{"destination_type": "city" or "country", "location_name": "<country or city name>", "cities": ["City1", "City2", ...]}}
"""

def weather_agent(state: TravelState):
    print("\nINSIDE WEATHER AGENT\n")

    query = state["user_query"]
    weather_data = None

    try:
        extraction_response = llm.invoke([
            SystemMessage(content="You are a precise travel query parser. Respond only with valid JSON."),
            HumanMessage(content=WEATHER_CITY_EXTRACTION_PROMPT.format(query=query))
        ])

        raw = extraction_response.content.strip()
        # Strip markdown code fences if the model adds them
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()

        parsed = json.loads(raw)
        cities = parsed.get("cities", [])

        print("\nEXTRACTED CITIES:", cities)

        if not cities:
            raise ValueError("No cities extracted from query")

        city_forecasts = asyncio.run(weather_mcp_multi_forecast(cities, days=5))

        print("\nWEATHER RESULTS:", city_forecasts)

        # Combine into one readable block for downstream prompts
        weather_data = "\n\n".join(
            f"--- {city} ---\n{forecast}" for city, forecast in city_forecasts.items()
        )

    except Exception as e:
        print(f"Weather information unavailable: {e}")
        weather_data = "Weather data unavailable."

    return {
        "weather_results": weather_data,
        "messages": [
            AIMessage(content="Weather forecast fetched for all destination cities.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ==========================
# Itinerary Agent
# ==========================


def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete multi-city travel itinerary if the destination is a country,
or a single-city itinerary if a specific city was requested.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Weather Forecast (per city):
{state['weather_results']}

Instructions:
- If multiple cities are covered in the weather forecast, split the itinerary across those cities
  in a sensible order (e.g. group nearby cities together, minimize backtracking).
- For each day, reference that city's actual forecasted weather and suggest weather-appropriate
  activities (e.g. indoor alternatives on rainy/stormy days, outdoor activities on clear days).
- Make the itinerary practical, budget-aware, and easy to follow.
- Include approximate travel time or transport between cities when the itinerary moves to a new city.
"""
    response = llm.invoke([
        SystemMessage(content = "You are an expert multi-city travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0)+1
    }



# ============================
# Final Response Agent
# ============================


def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Weather Forecast:
{state.get('weather_results', 'Not available')}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Forecast
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Reference the weather forecast when relevant to packing or activity suggestions.
- Keep the response useful for real travel planning.
"""

    
# =======================
# Build Graph
# =======================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =======================
# PostgreSQL Checkpointer
# =======================

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit = True,
    row_factory = dict_row
)

checkpointer = PostgresSaver(_conn) # checkpointer stores all the progress in the memory.
checkpointer.setup() 

travel_graph = graph.compile(checkpointer = checkpointer) # adds the graph as the blueprint into the memory.


# =======================
# function for FastAPI
# =======================


def run_travel_agent(user_input:str, thread_id:str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(  # follows the blueprint graph for the user input.
        {
            "messages": [
                HumanMessage(content = user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config = config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }