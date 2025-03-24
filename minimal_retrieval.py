import faiss
from faiss import Index
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from contextlib import asynccontextmanager
import time
import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import numpy as np


class ServerConfig(BaseSettings):
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHUNK_PATH: str = f"./data/chunks.npy"
    INDEX_PATH: str = f"./data/index_flat.faiss"


cfg = ServerConfig()

index: Index
model: SentenceTransformer
chunks: np.ndarray


class QueryRequest(BaseModel):
    prompt: str
    doc_no: int


def retrieve_docs(request: QueryRequest):
    global index, model, chunks
    query = model.encode([request.prompt], convert_to_numpy=True)
    _, ids = index.search(query, request.doc_no)
    if np.all((ids >= 0) & (ids < len(chunks))):
        return [chunks[i] for i in ids[0]]
    else:
        print(f"Some ids are out of range <{len(chunks)} : {ids}")
        return []


def init_app():
    global model, index, chunks, cfg
    faiss.cvar.distance_compute_blas_threshold = 1
    print("Loading Model...")
    t0 = time.perf_counter()
    model = SentenceTransformer(cfg.MODEL_NAME, device="cpu")
    t1 = time.perf_counter()
    print(f"Loaded {cfg.MODEL_NAME} model in {t1-t0} sec")
    index = faiss.read_index(cfg.INDEX_PATH)
    t2 = time.perf_counter()
    print(f"Loaded {os.path.basename(cfg.INDEX_PATH)} Index in {t2-t1} sec")
    chunks = np.load(cfg.CHUNK_PATH, allow_pickle=True)
    t3 = time.perf_counter()
    print(f"Loaded {os.path.basename(cfg.CHUNK_PATH)} Chunk in {t3-t2} sec")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_app()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {
        "status": "Server is running",
    }


@app.post("/retrieve")
async def retrieval_single_query(request: QueryRequest):
    result = retrieve_docs(request)
    return {"data": result}


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6000)


if __name__ == "__main__":
    main()
