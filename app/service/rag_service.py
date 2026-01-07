from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from app.util import pdf_util

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def ingest_documents_in_vector_store(user_id: str):
    pdf_texts: dict = dict()
    for filename, pdf_bytes in pdf_util.read_pdfs_as_bytes("./docs"):
        print(filename, len(pdf_bytes))
        pdf_texts[filename] = pdf_util.extract_text_from_pdf(pdf_bytes)

    chunks = pdf_util.split_multiple_pdfs(pdf_texts)
    create_vector_store(user_id, chunks)

def create_vector_store(user_id, chunks):
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=[{"user_id": user_id}] * len(chunks)
    )
    vectorstore.save_local(f"vectorstore/{user_id}")

def load_vector_store(user_id: str):
    return FAISS.load_local(
        f"vectorstore/{user_id}",
        embeddings,
        allow_dangerous_deserialization=True
    )

def query_data(user_id, question):

    # Step #1 Create embedding and save the vectors in in-memory store. In real world scenario,
    #         this step is a separate activity. An ETL pipeline (i.e., Spark or Dataflow) will be
    #         responsible for ingesting the embedding data into vector stores like Pinecone, Alloy DB etc.,
    ingest_documents_in_vector_store(user_id)
    # Step #2 Get reference to in-memory vector store
    vectorstore = load_vector_store(user_id)
    # Step #3 Create a retriever object (Vector similarity search) return 4 documents,
    #         only from documents where metadata["user_id"] == user_id
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"user_id": user_id}
        }
    )
    # Step #4 Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    # Step # 5 Call the chain with the question
    answer = qa_chain.invoke(question)
    return answer