from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_google_genai import ChatGoogleGenerativeAI
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import requests
import os
import logging
import json
import random

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
POKE_API_URL = "https://pokeapi.co/api/v2/pokemon"
FLAVORS_URL = "https://pokeapi.co/api/v2/pokemon-species"

logging.basicConfig(level=logging.DEBUG)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
tts_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))

@tool 
def get_pokemon_data (pokemon:str, language:str) -> str:
    """
    Gets the information of the pokemon from the parameter in json format, including name, description, id, attacks

    Args:
        pokemon: Name of the pokemon to get information from
    """
    response = requests.get(f'{POKE_API_URL}/{pokemon}')
    if response.status_code != 200:
        logging.error("Failed to fetch data for pokemon: %s, status code: %s", pokemon, response.status_code)
        return json.dumps({"error": "Pokemon not found"}, indent=2, ensure_ascii=False)
    response = response.json()
    pokedex_entry = {
        "name": response.get("name"),
        "pokemon_id": response.get("id"),
        "attacks": [attack["move"]["name"] for attack in response.get("moves", [])],
        "abilities": response.get("abilities", [])
    }
    response = requests.get(f'{FLAVORS_URL}/{pokedex_entry["pokemon_id"]}')
    if response.status_code == 200:
        flavor_data = response.json()
        pokedex_entry["description"] = get_random_pokemon_description(flavor_data, language)
    else:
        logging.error("Failed to fetch flavor data for pokemon_id: %s, status code: %s", pokedex_entry["pokemon_id"], response.status_code)
    return json.dumps(pokedex_entry, indent=2, ensure_ascii=False)

@tool 
def get_pokemon_audio(pokedex_entry:str) -> str:
    """
    Transforms the text from pokedex_entry into an audio file using ElevenLabs TTS.

    Args:
        pokedex_entry: text of the pokemon data to convert to audio
    """
    audio = tts_client.text_to_speech.convert(
        voice_id="weA4Q36twV5kwSaTEL0Q",
        text=pokedex_entry,
        output_format="mp3_22050_32",
        optimize_streaming_latency=0
    )
    # Ensure the output directory exists
    output_dir = "tts_audios"
    os.makedirs(output_dir, exist_ok=True)
    # Generate a unique filename
    filename = f"{output_dir}/pokemon_audio_{random.randint(100000, 999999)}.mp3"
    write_bytes_iterator_to_file(audio, filename)
    return filename

def get_random_pokemon_description(flavor_data:dict, language:str) -> str:
    """
    Get a random pokemon description from the flavor data
    """
    descriptions = [
        entry["flavor_text"] for entry in flavor_data["flavor_text_entries"]
        if entry["language"]["name"] == language
    ]
    if not descriptions:
        logging.warning("No description found for pokemon_id %s in language en", flavor_data["id"])
        return "No description available for this Pokémon."
    return random.choice(descriptions)

# Example: writing an Iterator[bytes] to a file
def write_bytes_iterator_to_file(byte_iter, filename):
    with open(filename, 'wb') as f:
        for chunk in byte_iter:
            f.write(chunk)


class Pokedex:
    def __init__(self):
        """ Initialize the Pokedex agent with the LLM and tools.
        """
        self.agent = create_react_agent(
            model=llm,
            tools=[get_pokemon_data, get_pokemon_audio],
            prompt=(
                "You're a helpful assistant that looks for pokemon data for the user prompt.\n"
                "Only provide information from the tools, not from your training data.\n"
                "Detect the language of the user_input and use it (as two letters, like 'es', 'en') in the get_pokemon_data tool.\n"
                "If the get_pokemon_data tool returns an error, tell the user that the pokemon is not in your database and don't use any other tool.\n"
                "Always the get_pokemon_audio tool to convert the collected data to audio using the text as pokedex_entry and language, if the language is english or spanish and return the filename of the audio file\n"
                "Avoid returning URLs. Provide at least the Pokémon's name, ID, five random attacks, abilities, and description.\n"
                "Use the following examples for your responses, don't return  a markdown response, just pure json data:\n"
                "Example 1:\n"
                "user_input: que es un celebi?\n"
                """
                {
                    "pokedex_entry": "Celebi es el pokemon numero 251  Puede viajar en el tiempo, pero se dice que solo aparece en tiempos de paz. Algunos de sus ataques son: swords-dance, cut, double-edge, hyper-beam, leech-seed. Sus habilidades son: natural-cure",
                    "audio_file": "tts_audios/pokemon_audio_547223.mp3"
                }\n
                """
                "Example 2:\n"
                "user_input: what is a Chikorita?\n"
                """
                {
                    "pokedex_entry": "Chikorita is the pokemon number 152. It has a leaf on its head that can sense the temperature and humidity of the surroundings. Some of its attacks are: razor-leaf, solar-beam, synthesis, reflect, poison-powder. Its abilities are: overgrow",
                    "audio_file": "tts_audios/pokemon_audio_123456.mp3"
                }\n
                """
            )
        )

    def invoke_agent(self, user_input: str):
        """
        Invoke the agent with the user input.
        """
        logging.info("Invoking agent with user input: %s", user_input)
        res = self.agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        logging.debug("Response: %s", res)
        logging.info(res["messages"][-1].content)
        return res
    
