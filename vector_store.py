from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

def create_vector_store(texts, embeddings):
    # create the vector store database
    db = FAISS.from_documents(texts, embeddings)
    return db

def save_vector_store(db):
    # save the vector store
    db.save_local("faiss_index")

def load_embeddings_model():
    #embeddings = load_embeddings_model_hf()
    embeddings = load_embeddings_model_openai()
    return embeddings

def load_vector_store():
    # load the vector store
    embeddings = load_embeddings_model()
    db = FAISS.load_local("faiss_index", embeddings)
    return db

def load_embeddings_model_hf():
    # load the embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    return embeddings

def load_embeddings_model_openai():
    # load the embeddings model
    embeddings = OpenAIEmbeddings()
    return embeddings