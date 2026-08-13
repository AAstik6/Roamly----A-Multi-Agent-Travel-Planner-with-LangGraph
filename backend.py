import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSl_CERT_FILE"] = certifi.where()
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
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

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


# ======================
# LLM
# ======================

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = GROQ_API_KEY
)

# =====================
# State
# =====================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add] # messages keep on appending by the reducer(operator.add())
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# =====================
# Flight Agent
# =====================

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content = "Flight results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0)+1
    }

# ===================
# Itinerary Agent
# ===================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""
    response = llm.invoke([
        SystemMessage(content = "You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0)+1
    }


# =======================
# Final Response Agent
# =======================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final trvel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""
    response = llm.invoke([
        SystemMessage(content = "You are a professional AI travel bbooking assistant."),
        HumanMessage(content = final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0)+1
    }