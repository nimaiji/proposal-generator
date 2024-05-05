import streamlit as st
import pandas as pd
import subprocess

from langchain.chains import LLMChain
# from langchain_community.llms import GPT4All
from gpt4all import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_falcon_template(idx):
    if idx == 0:
        template = """### USER: {target} describe their business with these sentences: {target_services} \n\n
        This is how {our} describe their services: {our_services} \n\n
        You must write a business proposal that convince {target} use {our}'s services.
        Also, you must proofread the whole business proposal.
        You must use estimations and facts to make your proposal more convincing.
        You must write only one paragraph.
        Do not write conclusion.
    
        ### Response: {section}: \n"""
    else:
        template = """### USER: write next section\n\n
    
        ### Response: {section}: \n"""

    return template

def get_llama_template():
    template = """Question: {target} describe their business with these sentences: {target_services} \n\n
    This is how {our} describe our services: {our_services} \n\n
    Based on collected information. write a proposal for {target}, that how {our} services can help their business.
    In this part just write {section} section of the proposal. Also, you must proofread the whole proposal before sending it to the client.
    You must use estimations and facts to make your proposal more convincing. You must only generate the output text nothing more. You are a Business Analyst writer.

    Answer: Let's accurately describe the business of {target} and how {our} services can help them."""

    return template

def get_mistral_template():
    template = """Question: {target} describe their business with these sentences: {target_services} \n\n
    This is how {our} describe our services: {our_services} \n\n
    Based on collected information. write a business proposal for {target}, that how {our} services can help their business.
    In this part just write {section} section of the proposal. Also, you must proofread the whole proposal before sending it to the client.
    You must use estimations and facts to make your proposal more convincing. You must only generate the output text nothing more. You are a Business Analyst writer.

    Answer: Let's accurately describe the business of {target} and how {our} services can help them."""

    return template


models = [
    {'path': './LLM/mistral-7b-openorca.gguf2.Q4_0.gguf'},
    {'path': './LLM/gpt4all-falcon-newbpe-q4_0.gguf'},
    {'path': './LLM/Meta-Llama-3-8B-Instruct.Q4_0.gguf'}
]

model = models[1]

callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model_name='gpt4all-falcon-newbpe-q4_0.gguf')


def get_services(input_data):
    services = pd.read_pickle(input_data)
    services = services.dropna()
    services = services[services['Class'] == 1]
    return services['Text'].tolist()[50:]


def generate_proposal(input_data, sections, our, our_services):
    services = get_services(input_data)
    target_services = ' '
    for service in services:
        target_services += service + '\n'
    target = 'Edinburgh Napier University'

    result = ''
    with llm.chat_session():
        for idx, section in enumerate(sections):
            if idx == 0:
                pmp = get_falcon_template(idx).format(target=target, our=our,
             target_services=target_services, our_services=our_services, section=section)
            else:
                pmp = get_falcon_template(idx).format(section=section)
            response = llm.generate(prompt=pmp, temp=0.5, max_tokens=2000)
            result += response + '\n\n'
    return result
    # prompt = PromptTemplate.from_template(model['template'])
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    # args_dict = {
    #     'target': target,
    #     'target_services': target_services,
    #     'our': our,
    #     'our_services': our_services,
    #     'section': section
    # }
    # return llm_chain.invoke(args_dict)

def show_proposal_generation_tab():
    st.header("Proposal Generation")
    our = st.text_input("Enter your company name:", "SmartVision")
    our_services_tmp = "SmartVision is an innovative online meeting platform company that revolutionizes virtual collaboration. With its cutting-edge technology, SmartVision offers a seamless and intuitive platform for businesses and individuals to connect, communicate, and collaborate remotely. Whether it's team meetings, client presentations, or virtual conferences, SmartVision provides acomprehensive suite of features, including high-definition video conferencing, screen sharing, real-time messaging, and advanced security protocols, ensuring a secure and productive virtual environment for all users. With a user-friendly interfaceand customizable options, SmartVision empowers organizations to enhance productivity, streamline communication, and foster meaningful connections in the digital age."
    our_services = st.text_area("Enter your company services:", our_services_tmp)

    input_data = st.text_input("Enter input data file path:", "./output.pickle")
    output_file = st.text_input("Enter output proposal file path:", "./output_proposal.pdf")

    if st.button("Generate Proposal"):
        if input_data:
            sections = ['Introduction', 'Context about the industry and market', 'Relevant trends or challenges',
                        'Timeline for completion', 'Benefits the company will gain from using services',
                        'Quantify any potential cost savings, efficiency improvements, or revenue increases']
            res = generate_proposal(input_data, sections, our, our_services)
            st.write(res)
        else:
            st.error("Please provide the input data file path.")

if __name__ == "__main__":
    show_proposal_generation_tab()
