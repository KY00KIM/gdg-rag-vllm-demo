import faiss
import asyncio
import queue
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from contextlib import asynccontextmanager
import time
import multiprocessing
from multiprocessing import Queue
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import  List

MP_CTX = "fork"
multiprocessing.set_start_method(MP_CTX)


class TServerConfig(BaseSettings):
    WORKDIR: str = "./data"
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHUNK_PATH: str = f"{WORKDIR}/chunk.npy"
    INDEX_PATH: str = f"{WORKDIR}/index_populate/wiki_1M_Flat.faiss"


cfg = TServerConfig()

index : faiss.Index 
model : SentenceTransformer
req_queue : Queue
res_queue : Queue


def retv_loop(q: Queue, res_q: Queue):
    print("[BACKGROUND]Initializing app...")
    faiss.cvar.distance_compute_blas_threshold = 1
    t0 = time.perf_counter()
    index = faiss.read_index(cfg.INDEX_PATH)
    t1 = time.perf_counter()
    print(f"[BACKGROUND] Load index{t1 -t0}")
    model = SentenceTransformer(cfg.MODEL_NAME)
    t2 = time.perf_counter()
    print(f"[BACKGROUND] Load model{t2 -t1}")
    while True:
        request = q.get(block=True)
        if "prompt" not in request.keys():
            print("ERROR : ", request)
            break
        t0 = time.perf_counter()
        query = model.encode(request["prompt"], convert_to_numpy=True)
        t1 = time.perf_counter()
        _, _ = index.search(query, request["doc_no"])
        t2 = time.perf_counter()
        print(f"[BACKGROUND] Took model {t1-t0:}sec  search {t2-t1} sec")
        res_q.put((t1 - t0, t2 - t1))


async def listen_loop(q: Queue):
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    print("START CONSUMER")
    while True:
        try:
            if loop.time() - start_time > 60:
                print("[CONSUMER] waiting for responses...")
                start_time = loop.time()
            # msg = q.get(block=False)
            msg = await loop.run_in_executor(None, q.get(block=True))
            print(f"[CONSUMER] -> {msg}")
            start_time = loop.time()
        except queue.Empty:
            await asyncio.sleep(0)
            continue


def init_app():
    global req_queue, res_queue
    req_queue = Queue(-1)
    res_queue = Queue(-1)
    p = multiprocessing.Process(
        target=retv_loop, args=[req_queue, res_queue], daemon=True
    )
    p.start()
    print("Start BACKGROUND")
    asyncio.create_task(listen_loop(res_queue))
    print("Start LISTENER")


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


class TQueryRequest(BaseModel):
    prompt: List[str]
    doc_no: int


@app.post("/retrieve/schedule")
async def retrieval_schedule(request: TQueryRequest):
    global index, model
    req_queue.put({"prompt": request.prompt, "doc_no": request.doc_no})
    return {"status": "done"}




def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="debug")


if __name__ == "__main__":
    main()
