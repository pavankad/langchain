from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
import os

def load_llama():
    """load the llm"""

    llm = CTransformers(model='/Users/pavan/models/llama-2-7b-chat.ggmlv3.q2_K.bin', # model available here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0})
    return llm

def load_openai():
    """load the llm"""

    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0, model_name="gpt-3.5-turbo", max_tokens=250
    ),

    return llm