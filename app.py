from click import prompt
import streamlit as st
import os
import tempfile
from document_assistant import DocumentAssistant

st.set_page_config(
    page_icon="📄",
    page_title="Document Assistant",
    layout="wide",    
)

#initialize the session variables
if 'assistant' not in st.session_state:
    st.session_state.assistant = DocumentAssistant()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store_loaded' not in st.session_state:
    st.session_state.vector_store_loaded = False
    
# app title and description
st.title("Document Assistant")
st.markdown(
    '''
    Upload your documents and ask questions about them. The assistant uses Retrieval-Augmented Generation (RAG) 
    to provide accurate answers based on your documents.
    '''
)

with st.sidebar:
    st.header('Document Mangement')
    
    uploaded_files = st.file_uploader(
        'Upload PDF Documents',
        type=['pdf'],
        accept_multiple_files=True,
    )
    
    if st.button('Process Documents') and uploaded_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_paths = []
            
            # Save uploaded files to temp directory
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, 'wb') as f:
                    f.write(file.getbuffer())
                temp_file_paths.append(file_path)
                
            # Process documents
            with st.spinner('Processing documents...'):
                st.session_state.assistant.process_documents(temp_file_paths)
                st.session_state.vector_store_loaded = True
                
            st.success(f'Processed {len(temp_file_paths)} documents.')
            
    # Save/load vector store options
    st.header('Vector Store Management')
    vector_store_dir = st.text_input('Vector Store Directory', 'document_vectors')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Save Vector Store') and st.session_state.vector_store_loaded:
            with st.spinner('Saving Vector Store..'):
                st.session_state.assistant.save_vector_store(vector_store_dir)
                st.success(f'Vector Store saved to {vector_store_dir}.')
                
    with col2:
        if st.button('Load Vector Store'):
            with st.spinner('Loading Vector Store'):
                st.session_state.assistant.load_vector_store(vector_store_dir)
                if st.session_state.assistant.vector_store:
                    st.session_state.vector_store_loaded = True
                    st.success(f'Vector Store Loaded')
                else:
                    st.error('Failed to Load Vector Store')
                    
# this is the main interface for asking questions
st.header('Ask Questions about your documents')

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.write(message['content'])
        if 'sources' in message and message['sources']:
            st.markdown('**Sources:**')
            for source in message['sources']:
                st.markdown(f'- {source}')
                

if prompt := st.chat_input('Ask question about ur Document'):
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    
    with st.chat_message('user'):
        st.write(prompt)
        
    with st.chat_message('assistant'):
        if not st.session_state.vector_store_loaded:
            st.write('Please Upload and Process Documents First, or load a saved Vector Store')
        else:
            with st.spinner('Thinking..'):
                response = st.session_state.assistant.answer_question(prompt)
                st.write(response['answer'])
                
                if 'sources' in response and response['sources']:
                    st.markdown('**Sources:**')
                    for source in response['sources']:
                        st.markdown(f'- {source}')
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response['answer'],
                'sources': response['sources'] if 'sources' in response else None
            })
            