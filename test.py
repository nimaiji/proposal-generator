from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

PATH = ('./LLM/Meta-Llama-3-8B-Instruct.Q4_0.gguf')
callbacks = [StreamingStdOutCallbackHandler()]

llm = GPT4All(model=PATH, callbacks=callbacks, verbose=True)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain = LLMChain(prompt=prompt, llm=llm)

res = llm_chain.run(question)
print('\n\n')
print(res)