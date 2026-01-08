import json
import os
from pathlib import Path
import traceback
from typing import Any, Dict
import uuid
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from memu.app import MemoryService

base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
chat_model = os.environ.get("CHAT_MODEL")
embed_model = os.environ.get("EMBED_MODEL")
postgres_dsn = os.environ.get("POSTGRES_DSN")

app = FastAPI()
service = MemoryService(
    llm_profiles = {
        "default": {
            "provider": "openai",
            "base_url": base_url,
            "api_key": api_key,
            "chat_model": chat_model,
            "embed_model": embed_model,
        },
    },
    database_config={
        "metadata_store": {
            "provider": "postgres",
            "ddl_mode": "create",
            "dsn": postgres_dsn,
        },
        "vector_index": {
            "provider": "pgvector",
            "dsn": postgres_dsn,
        },
    },
    retrieve_config={"method": "rag"},
)

storage_dir = Path(os.getenv("MEMU_STORAGE_DIR", "./data"))
storage_dir.mkdir(parents=True, exist_ok=True)

@app.post("/memorize")
async def memorize(payload: Dict[str, Any]):
    file_path = storage_dir / f"conversation-{uuid.uuid4().hex}.json"
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        result = await service.memorize(resource_url=str(file_path), modality="conversation")
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(str(file_path)):
            os.remove(str(file_path))

@app.post("/retrieve")
async def retrieve(payload: Dict[str, Any]):
    if "query" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    try:
        result = await service.retrieve([payload["query"]])
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
async def root():
    return {"message": "Hello MemU user!"}
