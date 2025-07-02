from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import requests
import os
import logging

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
POKE_API_URL = "https://pokeapi.co/api/v2/pokemon"
FLAVORS_URL = "https://pokeapi.co/api/v2/pokemon-species"

logging.basicConfig(level=logging.INFO)



llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

@tool
def get_pokemon_data (pokemon:str) -> str:
    """
    Gets the information of the pokemon from the parameter in json format, including name, description, id, attacks

    Args:
        pokemon: Name of the pokemon to get information from
    """
    response = requests.get(f'{POKE_API_URL}/{pokemon}')
    return response.json()

@tool
def get_pokemon_description(pokemon_id:int) -> str:
    """
    Get's the pokemon description for the pokemon_id on different languages

    Args:
        pokemon_id: id of the pokemon to get description from
    """

    response = requests.get(f'{FLAVORS_URL}/{pokemon_id}')
    return response.json()

agent = create_react_agent(
    model=llm,
    tools=[get_pokemon_data,get_pokemon_description],
    prompt=(
        "You're a helpful assistant that will return any json data on a human readable format"
        "If the user needs the pokemon's description then get the pokemon data first, then"
        "use the id from the json to get the pokemon description in the language of the original prompt" 
        "Avoid returning links in the response by just ignoring that part of the prompt"
    )
)

res = agent.invoke(
    {"messages": [{"role": "user", "content": "what pokemon is pichu? give me its description, its ID, at least 5 random attacks, and locations"}]}
)

logging.debug("Response: %s", res)
for msg in reversed(res["messages"]):
    if msg.__class__.__name__ == "AIMessage" and msg.content:
        print("AI:", msg.content)
        break
