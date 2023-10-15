import os
import streamlit as st

from PIL import Image
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from html_template import css, bot_template, user_template

load_dotenv('.env')


def get_chunks(pdf_docs=None):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separator="\n",
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(documents):

    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # Convert the document chunks to embedding and save them to the vector store
    vectordb = Chroma.from_texts(
        documents,
        embedding=embeddings,
        persist_directory="./vector_db"
    )
    vectordb.persist()
    return vectordb


def get_conversation_chain(vectordb):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        retriever=vectordb.as_retriever(),
        verbose=False,
        # return_source_documents=True,
        memory=memory
    )
    return conversation_chain


def handle_userinput(prompt):
    response = st.session_state.conversation({'question': prompt})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message('user', avatar=Image.open('./data/user_logo.png')):
                st.write(message.content, unsafe_allow_html=True)

        else:
            with st.chat_message('assistant', avatar=Image.open('./data/bot_image.png')):
                st.write(message.content, unsafe_allow_html=True)


def is_folder_empty(folder_path):
    # Use os.listdir to get a list of all items in the folder
    items = os.listdir(folder_path)

    # Check if the list of items is empty
    if len(items) == 0:
        return True
    else:
        return False


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    prompt = st.chat_input("Ask a question about your documents:")

    if prompt:
        handle_userinput(prompt)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                if is_folder_empty(folder_path="./vector_db"):
                    # get the text chunks
                    text_chunks = get_chunks(pdf_docs)
                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    print("create vector store")
                else:
                    # load vector store
                    print("Load existing vector store")
                    embeddings = OpenAIEmbeddings()
                    vectorstore = Chroma(
                        persist_directory="./vector_db", embedding_function=embeddings)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == "__main__":
    main()
