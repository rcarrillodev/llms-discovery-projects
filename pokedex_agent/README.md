# 🧑‍💻 Pokedex Agent 🎙️

A conversational agent that provides detailed Pokémon information and generates text-to-speech audio responses using [LangChain](https://python.langchain.com/), [Google Gemini](https://ai.google.dev/gemini-api/docs/get-started), and [ElevenLabs TTS](https://elevenlabs.io/).  
Gotta catch 'em all — with your ears! 🎧

## ✨ Features

- 🔎 Retrieves Pokémon data (name, ID, description, attacks, abilities) from the [PokeAPI](https://pokeapi.co/).
- 🌐 Detects the user's language (English or Spanish) and responds accordingly.
- 🔊 Converts Pokémon information into audio using ElevenLabs TTS.
- 📝 Returns responses in a human-readable format, with audio file output for supported languages.
- 🌐 **Exposes a FastAPI endpoint for programmatic access.**

## ⚡ Requirements

- 🐍 Python 3.8+
- API keys for:
  - 🔑 Google Gemini (`GOOGLE_API_KEY`)
  - 🔑 ElevenLabs (`ELEVEN_LABS_API_KEY`)

## 🛠️ Installation

1. Clone the repository.
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Create a `.env` file in the root directory with your API keys:
    ```
    GOOGLE_API_KEY=your_google_api_key
    ELEVEN_LABS_API_KEY=your_elevenlabs_api_key
    ```

## 🚀 Usage

Start the FastAPI server:

```sh
uvicorn main:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000).

### Query the Endpoint

Send a GET request to `/whosthatpokemon` with a JSON body containing your message:

**Example request:**
```sh
curl -X GET "http://localhost:8000/whosthatpokemon" -H "Content-Type: application/json" -d '{"message": "que es un bulbasaur?"}'
```

**Example response:**
```json
{
  "pokedex_entry": "Bulbasaur es el Pokémon número 1. Es un Pokémon de tipo planta/veneno. Algunos de sus ataques son: tackle, vine-whip, razor-leaf, growl, leech-seed. Sus habilidades son: overgrow, chlorophyll.",
  "audio_file": "tts_audios/pokemon_audio_123456.mp3"
}
```

🎵 Audio files are saved in the `tts_audios/` directory.

## 📁 Project Structure

- `main.py` — FastAPI app exposing the agent as an endpoint.
- `pokedex_agent/pokedex.py` — Main script with agent logic and tool definitions.
- `tts_audios/` — Directory for generated audio files.
- `.env` — Environment variables for API keys.

## 🧩 Example Output

```
{
  "pokedex_entry": "Bulbasaur es el Pokémon número 1. Es un Pokémon de tipo planta/veneno. Algunos de sus ataques son: tackle, vine-whip, razor-leaf, growl, leech-seed. Sus habilidades son: overgrow, chlorophyll.",
  "audio_file": "tts_audios/pokemon_audio_123456.mp3"
}
```

An audio file (e.g., `tts_audios/pokemon_audio_123456.mp3`) will also be generated. 🎶

## 📜 License

[MIT License](LICENSE)

---

Made with ❤️ by RCDev