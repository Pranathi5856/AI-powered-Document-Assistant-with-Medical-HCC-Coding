from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, load_summarize_chain

def setup_retriever(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

def answer_question(retriever, question):
    llm = ChatGroq(
        model="llama3-8b-8192",  # Or try "mixtral-8x7b-32768"
        temperature=0.2
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": question})
    return result["result"]

def summarize_document(documents):
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.2
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain({"input_documents": documents}, return_only_outputs=True)
    return summary["output_text"]