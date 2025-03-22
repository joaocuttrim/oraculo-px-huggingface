# app.py
import streamlit as st
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from loaders.load_file import load_file

st.set_page_config(page_title="Oráculo PX - HuggingFace", layout="wide")
st.title("Oráculo PX - Gerenciamento de Projetos (modelo gratuito via HuggingFace)")

huggingface_api_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if not huggingface_api_key:
    st.error("Chave da API do HuggingFace não encontrada. Verifique o secrets.toml.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

uploaded_file = st.sidebar.file_uploader("Envie um arquivo (PDF, CSV ou TXT)", type=["pdf", "csv", "txt"])

if uploaded_file:
    documents = load_file(uploaded_file)
    if documents is None:
        st.error("Erro ao carregar o arquivo.")
        st.stop()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    chain = load_qa_chain(llm, chain_type="stuff")

    query = st.text_input("Faça sua pergunta sobre o arquivo carregado:")

    if query:
        resposta = chain.run(input_documents=docs, question=query)
        st.write("### Resposta:")
        st.write(resposta)
else:
    st.info("Por favor, envie um arquivo para começar.")
