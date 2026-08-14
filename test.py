from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent

# res = tavily_search("give me the best hotels in India.")
# print(res)

#res = search_flights("Plan a 7 days Japan trip from India")
#print(res)

user_input = input("enter the request:")

response = run_travel_agent(
  user_input = user_input,
  thread_id = "test_123"
)
print("\nFINAL RESPONSE:\n")
print(response["answer"])

