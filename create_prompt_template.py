from langchain import PromptTemplate

def create_prompt_template():
    # prepare the template that provides instructions to the chatbot

    template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".
    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'])
    return prompt