# RAG Demo

### Requirement

- `docker`

## 1. vLLM Build & Run

```bash
git clone https://github.com/vllm-project/vllm
cd vllm

# If x86
docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .

# If aarch(AppleSillicon)
docker build -f Dockerfile.arm -t vllm-cpu-env --shm-size=4g .
```

```**bash**
docker run -it --rm --network=host vllm-cpu-env  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# "facebook/opt-125m"
```

## 2. Retrieval Build & Run

- Configure `data/input.txt` as knowledge base
  - delim docs with `'\n'` newline character

```bash
git clone https://github.com/KY00KIM/gdg-rag-vllm-demo
cd gdg-rag-vllm-demo
docker build -f Dockerfile -t retrieval:0.1 .
```

```bash
docker run -it  --rm  -p 6000:6000  -v "$(pwd)/data:/app/data"  retrieval:0.1

# Configure data/input.txt as Knowledge Base
```

```bash
# Inside container shell,
# Bootstrap(Embedding, Chunk, Index)
python embed.py --type bootstrap

# Test bootstrap output
python embed.py --type test
# Exit container
```

```bash
# Run retrieval server
# -d # Optional for daemon
docker run -it --rm -p 6000:6000 -v "$(pwd)/data:/app/data" retrieval:0.1  python minimal_retrieval.py
```

## 3. Run RAG Chat Client

```bash
# By Local
pip install openai requests
python client.py

# By Container
docker run -it --rm --network=host -v "$(pwd)/data:/app/data" retrieval:0.1  python client.py

# If update client.py
docker cp "$(pwd)/client.py" $CONTAINER_ID:/app
```
