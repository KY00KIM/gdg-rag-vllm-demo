FROM python:3.10.16-bullseye

WORKDIR /app

COPY minimal_retrieval.py .
COPY embed.py .

RUN apt-get update -y &&\
  apt-get install -y vim &&\
  pip install \
  faiss-cpu \
  fastapi \
  sentence-transformers \
  uvicorn \
  pydantic-settings

RUN python -c "from sentence_transformers import SentenceTransformer;SentenceTransformer('all-MiniLM-L6-v2')"


CMD ["/bin/bash"]
