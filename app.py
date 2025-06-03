import streamlit as st
from document_loader import load_document
from langchain_pipeline import setup_retriever, answer_question, summarize_document
from hcc_mapping import load_hcc_codes, extract_diagnoses, map_to_hcc_codes
from dotenv import load_dotenv
load_dotenv()

st.title("AI-powered Document Assistant with Medical HCC Coding")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = load_document(uploaded_file.name)
    retriever = setup_retriever(docs)
    st.success("Document loaded.")

    doc_text = "\n".join([d.page_content for d in docs])

    # Medical Document HCC Extraction
    if st.checkbox("Analyze for HCC codes (medical docs)"):
        hcc_df = load_hcc_codes()
        diagnoses = extract_diagnoses(doc_text)
        mappings = map_to_hcc_codes(diagnoses, hcc_df)
        if mappings:
            st.write("**Identified Diagnoses and HCC Codes:**")
            for m in mappings:
                st.write(f"- {m['diagnosis'].title()} → {m['hcc_code']} ({m['description']})")
        else:
            st.write("No standard diagnoses found for HCC mapping.")

    # Q&A
    query = st.text_input("Ask a question about your document:")
    if query:
        answer = answer_question(retriever, query)
        st.write("**Answer:**", answer)

    # Summarization
    if st.button("Summarize Document"):
        summary = summarize_document(docs)
        st.write("**Summary:**", summary)