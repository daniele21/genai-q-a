import streamlit as st
import random
import time

from scripts.agent_init import init_agent
from scripts.llm_load import load_llms

st.set_page_config(
    page_title="RAG Example", 
    initial_sidebar_state="collapsed", 
    layout="wide",
    #page_icon='image/assistant_logo.webp',
)

## Load CSS ##
@st.cache_data
def load_css(file_name = "./scripts/css/style.css"):
    with open(file_name) as f:
        css = f'<style>{f.read()}</style>'
    return css
css = load_css()
st.markdown(css, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col2:
    st.title("💬 RAG Example")

#with col3:
#    st.image('image/assistant_logo.webp', width=100)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages =  [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        with st.spinner():

            llm_path = '/Users/moltisantid/.cache/lm-studio/models/' + 'TheBloke/OpenHermes-2.5-Mistral-7B-16k-GGUF/openhermes-2.5-mistral-7b-16k.Q5_K_M.gguf'
            split_params = {'chunk_size': 100,
                            'chunk_overlap': 20}
            emb_path = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
            memory_params = {'max_token_limit': 40}
            agent = init_agent(llm_path, split_params, emb_path, memory_params)
            response = agent.invoke(prompt)
            #print(response['message']['content'])

            with st.chat_message("assistant"):
                st.markdown( response['output'])
            st.session_state.messages.append({"role": "assistant", "content": response['output']})
       # st.session_state.chat_message("assistant").write('asnwer')

        