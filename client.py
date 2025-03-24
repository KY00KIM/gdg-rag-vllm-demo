from openai import OpenAI
from typing import List
import requests

OPENAI_API_KEY = "EMPTY"  # 더미 값, 필요 X
OPENAI_API_BASE_URL = "http://localhost:8000/v1"  # 서빙할 vLLM 서버 주소
RETRIEVAL_URL = "http://localhost:6000/retrieve"  # 벡터 DB 서버 주소


def apply_rag_template(msg: str, documents: List[str]):
    """
    문서와 사용자 메시지에 RAG 템플릿을 적용한 하나의 쿼리를 반환합니다.
    템플릿 변경.

    Args:
        msg (str): The user's message or question.
        documents (List[str]): A list of documents to refer to.

    Returns:
        str: The formatted message with the RAG template applied.
    """
    template = """You are Helpful AI assistant. refer to documents in [Ref], ease answer the user’s question kindly. Do not say anything that is not in the document, and if the information is not in the document, please say that it is not available. And just respond with answers nothing else.
    Question: {question}
    [Ref]
    {ref}
    Answer: """

    reference = "\n".join([doc for doc in documents])

    return template.format(question=msg, ref=reference)


def get_documents(msg: str, doc_no: int):
    """
    (개발중)더미 문서 Retrieval 함수
    벡터 DB 연동해서 관련 문서 가져올 예정..

    Args:
        msg (str): The message to retrieve documents for.
        doc_no (int): The number of documents to retrieve.

    Returns:
        List[str]: A list of related documents.
    """
    response = requests.post(RETRIEVAL_URL, json={
                             "prompt": msg, "doc_no": doc_no})
    if response.status_code != 200:
        print(f"Failed to retrieve documents")
        return []
    return response.json()["data"]


if __name__ == "__main__":
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE_URL,
    )
    USER_MESSAGE = "Who were leader of GDG on campus Korea University?"
    MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    NO_DOCUMENTS = 2  # 벡터 DB로부터 가져올 관련 문서 개수

    STREAM = True
    DO_RAG = True
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        message = input("Enter message: ")
        if message == "quit":
            break
        else:
            if DO_RAG:
                documents = get_documents(message, NO_DOCUMENTS)
                message = apply_rag_template(message, documents)
            messages.append({"role": "user", "content": message})
            # print("User: ", augmented_message)
            if STREAM:
                chat_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True,
                    extra_body={
                        "top_k": 1,
                        "temperature": 0.0,
                    }
                )
                response = ""
                print("Response: ", end='', flush=True)
                for chunk in chat_response:
                    delta_content = chunk.choices[0].delta.content
                    print(delta_content, end='', flush=True)
                    response += delta_content
            else:
                chat_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=False,
                    extra_body={
                        "top_k": 1,
                        "temperature": 0.0,
                    }
                )
                response = chat_response.choices[0].message.content
                print(
                    chat_response.choices[0].message.content, end='', flush=True)
            print()
            messages.append(
                {"role": "assistant", "content": response})
