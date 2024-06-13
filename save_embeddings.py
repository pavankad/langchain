import pdb

from langchain.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader
import os
from langchain_community.document_loaders import ConfluenceLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import create_vector_store, save_vector_store, load_embeddings_model

def load_confluence_docs():
    loader = ConfluenceLoader(
        url="https://pavan-team.atlassian.net/wiki/", username="pavankad@gmail.com", api_key=os.environ['ATLASSIAN_API_KEY'])
    documents = loader.load(space_key="TD", include_attachments=True, limit=50)
    return documents

def load_csv_files():
    loader = CSVLoader()
    documents = loader.loadr()
    return documents
def load_pdf_docs():
    # load the pdf files
    loader = DirectoryLoader(path='/Users/pavan/docs', glob="*.pdf", loader_cls=PyPDFLoader)
    # split the documents into chunks
    documents = loader.load()
    return documents

def split_doc_txt(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                              chunk_overlap=50)
    texts = splitter.split_documents(documents)
    return texts

def load_embeddings(source):
    pdb.set_trace()
    if source =='pdf':
        documents = load_pdf_docs()
    elif source == 'confluence':
        documents = load_confluence_docs()
    elif source == 'csv':
        document = load_csf_files()
    texts = split_doc_txt(documents)
    pdb.set_trace()
    embeddings = load_embeddings_model()
    db=create_vector_store(texts, embeddings)
    save_vector_store(db)
    pdb.set_trace()
    print("hello1")

if __name__== "__main__":
    source = "pdf"
    load_embeddings(source)
