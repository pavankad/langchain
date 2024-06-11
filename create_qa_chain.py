import pdb

from langchain.chains import RetrievalQA
from create_prompt_template import create_prompt_template
from vector_store import load_vector_store
from llm import load_llama
from llm import load_openai

def create_qa_chain():
    """create the qa chain"""
    # load the llm, vector store, and the prompt
    llm = load_llama()
    #llm = load_openai()
    db = load_vector_store()
    prompt = create_prompt_template()

    # create the qa_chain
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain

def generate_response(query, qa_chain):

    # use the qa_chain to answer the given query
    return qa_chain({'query':query})['result']