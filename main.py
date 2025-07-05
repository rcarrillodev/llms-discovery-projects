from fastapi import FastAPI, Request
from pydantic import BaseModel
from pokedex_agent.pokedex import Pokedex
import logging
import re
import json

app = FastAPI()
pokedex = Pokedex()

class QueryRequest(BaseModel):
    message: str

@app.get("/whosthatpokemon")
async def ask_pokedex(query: QueryRequest):
    try:
        logging.info("Received query: %s", query.message)
        res = pokedex.invoke_agent(query.message)
        answer = res["messages"][-1].content

        # Extract JSON from Markdown code block if present
        match = re.search(r"```json\s*(\{.*?\})\s*```", answer, re.DOTALL)
        if match:
            answer_json = json.loads(match.group(1))
            return answer_json
        # If not in code block, try to parse directly
        try:
            return json.loads(answer)
        except Exception:
            return {"result": answer}
    except Exception as e:
        logging.exception("Error processing request")
        return {"error": str(e)}