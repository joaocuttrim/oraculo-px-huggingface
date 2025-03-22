# app.py
import streamlit as st
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from loaders.load_file import load_file

st.set_page_config(page_title="Oráculo PX - HuggingFace (Falcon)", layout="wide")
st.title("Oráculo PX - Gerenciamento de Projetos (modelo Falcon com prompt melhorado)")

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

    # Normaliza o conteúdo
    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", " ").strip()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )

    # Prompt melhorado
    prompt_template = PromptTemplate(
        template="""Use the following pieces of context to answer the question at the end.
If the answer cannot be found in the context, just say "Não sei" — do not make up an answer.

{context}

Pergunta: {question}
Resposta útil:""",
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

    query = st.text_input("Faça sua pergunta sobre o arquivo carregado:")

    if query:
        resposta = chain.run(input_documents=docs, question=query)
        st.write("### Resposta:")
        st.write(resposta)
else:
    st.info("Por favor, envie um arquivo para começar.")
