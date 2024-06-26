import os
import pdb

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

pdb.set_trace()


chatbot = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0, model_name="gpt-3.5-turbo", max_tokens=500
    ),
    chain_type="stuff",
    retriever=FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True ).as_retriever(search_type="similarity", search_kwargs={"k": 2})
)

template = """
Please enlist points and provide details. {query}?
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)
"""
print(chatbot.run(
    prompt.format(query="what are important features of dynamoDB?")
))

print(chatbot.run(
    prompt.format(query="Compare LSM and B-Tree")
))

"""

print(chatbot.run(
    prompt.format(query="What is the gist of US health care mentioned in the doc?")
))
# --v is a parameter used to specify a specific model version in Midjourney's AI image generation tool.