from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_google_genai import ChatGoogleGenerativeAI
from elevenlabs import ElevenLabs, VoiceSettings
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

logging.basicConfig(level=logging.INFO)



llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
tts_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))

@tool
def get_pokemon_data (pokemon:str) -> str:
    """
    Gets the information of the pokemon from the parameter in json format, including name, description, id, attacks

    Args:
        pokemon: Name of the pokemon to get information from
    """
    response = requests.get(f'{POKE_API_URL}/{pokemon}').json()
    data = {
        "name": response.get("name"),
        "pokemon_id": response.get("id"),
        "attacks": [attack["move"]["name"] for attack in response.get("moves", [])],
        "abilities": response.get("abilities", [])
    }
    return json.dumps(data, indent=2, ensure_ascii=False)

@tool
def get_pokemon_description(pokemon_id:int, language:str) -> str:
    """
    Get's the pokemon description for the pokemon_id on different languages, the language is the one used in the original prompt
    using two letters (es, en, fr, etc)

    Args:
        pokemon_id: id of the pokemon to get description from
        language: language code for the description (es, en, fr, etc)
    """

    response = requests.get(f'{FLAVORS_URL}/{pokemon_id}')
    data = response.json()
    logging.debug("language: %s", language)
    descriptions = [
        entry["flavor_text"] for entry in data["flavor_text_entries"]
        if entry["language"]["name"] == language
    ]
    if not descriptions:
        logging.warning("No description found for pokemon_id %s in language %s", pokemon_id, language)
        return data
    return json.dumps(descriptions[random.randint(0, len(descriptions) - 1)], indent=2, ensure_ascii=False)

@tool
def get_pokemon_audio(pokedex_entry:str, language:str) -> str:
    """
    Transforms the text from pokemon_entry into audio, the language is the one used in the original prompt
    using two letters (es, en, fr, etc)

    Args:
        pokedex_entry: text of the pokemon data to convert to audio
        language: language code for the audio (es, en, fr, etc)
    """
    audio = tts_client.text_to_speech.convert(
        voice_id="weA4Q36twV5kwSaTEL0Q",
        text=pokedex_entry,
        output_format="mp3_44100_128"
    )
    # Ensure the output directory exists
    output_dir = "tts_audios"
    os.makedirs(output_dir, exist_ok=True)
    # Generate a unique filename
    filename = f"{output_dir}/pokemon_audio_{random.randint(100000, 999999)}.mp3"
    write_bytes_iterator_to_file(audio, filename)
    return filename

# Example: writing an Iterator[bytes] to a file
def write_bytes_iterator_to_file(byte_iter, filename):
    with open(filename, 'wb') as f:
        for chunk in byte_iter:
            f.write(chunk)

agent = create_react_agent(
    model=llm,
    tools=[get_pokemon_data,get_pokemon_description, get_pokemon_audio],
    prompt=(
        "You're a helpful assistant that will return any json data on a human readable format, provide information only from the tools not from your training data"
        "Detect the language of the user prompt and pass it as the language parameter to the get_pokemon_description tool in form of two letters (es, en, fr, etc)"
        "If the user needs the pokemon's description then get the pokemon data first, then"
        "use the id from the json to get the pokemon description in the language of the original prompt" 
        "Avoid returning links in the response by just ignoring that part of the prompt"
        "provide at least the pokemon's name, id, attacks and abilities and descripion, don't list the data, return the information on a single paragraph"
        "Use the get_pokemon_audio tool to convert the collected data to audio using the text as pokedex_entry and language, if the language is en or es and return the filename of the audio file"
    )
)

res = agent.invoke(
    {"messages": [{"role": "user", "content": "que es un celebi?"}]}
)

logging.debug("Response: %s", res)
for msg in reversed(res["messages"]):
    if msg.__class__.__name__ == "AIMessage" and msg.content:
        print("AI:", msg.content)
        break
