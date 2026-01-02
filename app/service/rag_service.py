from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def load_vector_store(user_id: str):
    return FAISS.load_local(
        f"vectorstore/{user_id}",
        embeddings,
        allow_dangerous_deserialization=True
    )


def query_data(user_id, question):
    vectorstore = load_vector_store(user_id)
    print("FAISS index size:", vectorstore.index.ntotal)
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"user_id": user_id}
        }
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    answer = qa_chain.invoke(question)
    return answer