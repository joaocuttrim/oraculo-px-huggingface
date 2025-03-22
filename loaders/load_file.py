# loaders/load_file.py
from langchain.document_loaders import TextLoader, CSVLoader, PyPDFLoader
import tempfile

def load_file(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if ext == "pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext == "csv":
        loader = CSVLoader(tmp_path)
    elif ext == "txt":
        loader = TextLoader(tmp_path)
    else:
        return None

    return loader.load()
