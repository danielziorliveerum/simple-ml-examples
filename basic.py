from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

model_name = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

config = GenerationConfig(max_new_tokens=200)
while True:
    query = input('Question: ')
    tokens = tokenizer(query, return_tensors='pt')
    print(tokens)
    outputs = model.generate(**tokens, generation_config=config)
    print(outputs)

    print(' '.join(tokenizer.batch_decode(outputs, skip_special_tokens=True)))