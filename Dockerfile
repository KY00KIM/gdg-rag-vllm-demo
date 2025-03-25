FROM python:3.10.16-bullseye

WORKDIR /app

COPY minimal_retrieval.py .
COPY embed.py .
COPY client.py .

RUN apt-get update -y &&\
  apt-get install -y --no-install-recommends vim &&\
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* &&\
  pip install --no-cache-dir \
  faiss-cpu \
  fastapi \
  sentence-transformers \
  uvicorn \
  pydantic-settings \ 
  openai \ 
  requests

RUN python -c "from sentence_transformers import SentenceTransformer;SentenceTransformer('all-MiniLM-L6-v2')"


CMD ["/bin/bash"]
