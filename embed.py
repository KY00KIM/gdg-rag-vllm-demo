import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/input.txt"
CHUNK_PATH = "data/chunks.npy"
EMBED_PATH = "data/embeddings.npy"
INDEX_PATH = "data/index_flat.faiss"


def bootstrap():
    # 1. Read input text
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip()
                 for line in f if line.strip()]  # remove empty lines

# 2. Save text as ndarray(chunks)
    chunks = np.array(lines, dtype=object)
    np.save(CHUNK_PATH, chunks)
    print(f"Saved {len(chunks)} texts to {CHUNK_PATH}")

# 3. Get embedding and save ndarray of texts
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        chunks, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMBED_PATH, embeddings)
    print(f"Saved embeddings to {EMBED_PATH}")

# 4. Make index and save
    dim = embeddings.shape[1]
    index = faiss.index_factory(dim, "Flat")
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index to {INDEX_PATH}")


def test():
    chunks = np.load(CHUNK_PATH, allow_pickle=True)
    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("RAG Search Ready. Type your query (or type 'exit' to quit):", flush=True)

    while True:
        query = input("Query > ")
        if query.strip().lower() == "exit":
            break

        query_embedding = model.encode([query])
        top_k = 1
        distances, indices = index.search(query_embedding, top_k)

        top_id = indices[0][0]
        print(f"Top Chunk ID: {top_id}")
        print(f"Matched Chunk: {chunks[top_id]}")
        print(f"Distance: {distances[0][0]:.4f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["bootstrap", "test"],
    )
    args = parser.parse_args()
    if args.type == "bootstrap":
        bootstrap()
    elif args.type == "test":
        test()
    else:
        print("Arg \'--type\' must be \'bootstrap\' or \'test\'")
