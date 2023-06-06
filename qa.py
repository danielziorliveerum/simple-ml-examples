from langchain.document_loaders import GoogleDriveLoader
from langchain.chains.question_answering import load_qa_chain
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

loader = GoogleDriveLoader(
    document_ids=['1wcKtMkchV6NoNPO6YUlkU51JCmG8GjZt70c4oumrBoo'],
)

documents = loader.load()

chain = load_qa_chain(llm=llm, chain_type='stuff')
while True:
    query = input('Question: ')
    print(chain.run(input_documents=documents, question=query))