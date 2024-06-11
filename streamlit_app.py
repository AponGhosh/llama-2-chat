import streamlit as st
import replicate
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# App title
st.set_page_config(page_title="🦙 Llama 2 Chatbot 💬")

# Replicate Credentials
with st.sidebar:
    st.title('🦙 Llama 2 Chatbot 💬')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API key:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your Replicate API key!', icon='⚠️')
        else:
            st.success('Successful! Ask your questions ...', icon='✅')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Load T5 model and tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Please feel free to ask your questions ..."}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Please feel free to ask your questions ..."}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":0.10, "top_p":0.90, "max_length":120, "repetition_penalty":1})
    return output

# Function for correcting grammatical errors using T5 model
def correct_grammar(text):
    input_text = "gec: " + text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = t5_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    corrected_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            
            # Correct the grammatical errors in the response
            corrected_response = correct_grammar(full_response)
            placeholder.markdown(corrected_response)
            
    message = {"role": "assistant", "content": corrected_response}
    st.session_state.messages.append(message)
