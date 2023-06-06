from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain, HuggingFaceHub

def local_model():
    model_name = 'google/flan-t5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048
    )

    return HuggingFacePipeline(pipeline=pipe)

llm = local_model()

template = """Context: {context}

Question: {question}

Answer: Let's think step by step.
"""

prompt = PromptTemplate(template=template, input_variables=["question", "context"])
chain = LLMChain(prompt=prompt,llm=llm,)
while True:
    query = input('Question: ')
    context = input('Context: ')
    print(chain.run({'question' : query, 'context' : context}))
    