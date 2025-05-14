import os
import tempfile
from typing import List,Dict
from urllib import response
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from sqlalchemy import all_
from torch import chunk

load_dotenv()

class DocumentAssistant:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.llm = ChatGoogleGenerativeAI(
            model = 'gemini-2.0-flash',
            temperature = 0
        )
        self.vector_store = None
        
    def process_documents(self, files: List[str]) -> None:
        '''Process Documents and build Vector Store'''
        all_chunks = []
        for file_path in files:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                # Split the documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 1000,
                    chunk_overlap = 200, #chunk_overlap = 200, the last 200 characters of one chunk will overlap with the first 200 characters of the next chunk.
                )
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                print(f'Processed {file_path} with {len(chunks)} chunks.')
            else:
                print(f'Unsupported file format: {file_path}')
                
        if all_chunks:
            #self.vector_store = FAISS.load_local(directory, self.embeddings, allow_dangerous_deserialization=True)
            self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
            print(f'Vector store built with {len(all_chunks)} chunks.')
        else:
            print('No valid documents to process.')
            
    def save_vector_store(self, directory:str) -> None:
        if self.vector_store:
            self.vector_store.save_local(directory)
            print(f'Vector Store saved to {directory}')
        else:
            print('No Vector Store to save.')
            
    def load_vector_store(self, directory:str) -> None:
        if os.path.exists(directory):
            self.vector_store = FAISS.load_local(directory, self.embeddings)
            print(f'Vector Store loaded from {directory}')
        else:
            print('No Vector Store found at the specified directory.')
        
    def answer_question(self, question:str) -> Dict:
        if not self.vector_store: 
            return {'answer': 'No vector store loaded. Please load a vector store first.'}
        
        # Create a RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = 'stuff',
            retriever = self.vector_store.as_retriever(search_kwargs={'k': 4}),
            #search_kwargs={'k': 4} means that for each query, the retriever will return the top 4 most similar document chunks.
            return_source_documents = True
        )
    
        result = qa_chain({'query': question})
        '''
        {
        'result': 'LangChain is a framework for developing applications powered by language models.',
        'source_documents': [
            Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={'source': 'doc1.pdf', 'page': 1}),
            Document(page_content="It provides tools for connecting LLMs to external data.", metadata={'source': 'doc2.pdf', 'page': 2}),
            # ... up to k documents
        ]
        }
        '''
        sources = []
        for doc in result.get('source_documents',[]):
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source']
                if source not in sources:
                    sources.append(source)
                    
        return {
            'answer': result['result'],
            'sources': sources
        }
        
        
def main():
    assistant = DocumentAssistant()
    vector_store_dir = 'document_vectors'
    
    if os.path.exists(vector_store_dir):
        assistant.load_vector_store(vector_store_dir)
    else:
        documents = ['path/to/your/document1.pdf']
        assistant.process_documents(documents)
        assistant.save_vector_store(vector_store_dir)
        
    print('Ready to answer questions. Type "exit" to quit.')
    
    while True:
        question = input('\n Your question: ')
        if question.lower() == 'exit':
            break
        response = assistant.answer_question(question)
        print(f'Answer: {response["answer"]}')
        
        if response.get('sources'):
            print('Sources:')
            for source in response['sources']:
                print(f'- {source}')
                

if __name__ == '__main__':
    main()