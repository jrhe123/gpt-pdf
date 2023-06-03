from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import os
import pickle
import pinecone
from tempfile import NamedTemporaryFile
from time import sleep
# langchain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# InstructorEmbedding
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceInstructEmbeddings

load_dotenv()
ENABLE_PINECONE = False
ENABLE_HUGGING_FACE = False  # comparison between openAI & hugging face
# initialize pinecone
# reference: https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/Chat_with_Any_Documents_Own_ChatGPT_with_LangChain.ipynb
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)


def store_embeddings(docs, embeddings, store_name):
    # FAISS (Facebook AI Similarity Search)
    vectorStore = FAISS.from_texts(docs, embeddings)
    # save it as cached
    with open(f"{store_name}.pk1", "wb") as f:
        pickle.dump(vectorStore, f)
    return vectorStore


def load_embeddings(store_name):
    with open(f"{store_name}.pk1", "rb") as f:
        print("loaded from cached")
        vectorStore = pickle.load(f)
    return vectorStore


def wait_on_index(index: str):
    """
    Takes the name of the index to wait for and blocks until it's available and ready.
    """
    ready = False
    while not ready:
        try:
            desc = pinecone.describe_index(index)
            print("!!!!!! check index status: ", desc)
            if desc[7]['ready']:
                return True
        except:
            pass
        sleep(5)


def main():

    # streamlit
    st.set_page_config(page_title="Ask your pdf")
    st.header("Ask your pdf")
    pdf = st.file_uploader("Upload your pdf here..", type="pdf")
    try:
        # read pdf
        if pdf is not None:
            if ENABLE_PINECONE:
                index_name = f"langchain-demo-{pdf.name[:-4]}"
                indexes = pinecone.list_indexes()
                embeddings = OpenAIEmbeddings()
                if index_name not in indexes:
                    # 1536 for openAI
                    pinecone.create_index(index_name, dimension=1536)
                    wait_on_index(index_name)
                    # upload vector to DB
                    with NamedTemporaryFile(dir='./temp', suffix='.pdf') as f:
                        f.write(pdf.getbuffer())
                        loader = PyPDFLoader(f.name)
                        data = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(
                            separators="\n",
                            chunk_size=1000, chunk_overlap=200)
                        documents = text_splitter.split_documents(data)
                        kownledge_base = Pinecone.from_documents(
                            documents, embeddings, index_name=index_name)
                        # kownledge_base = Pinecone.from_texts(
                        #     [t.page_content for t in documents], embeddings, index_name=index_name)
                else:
                    kownledge_base = Pinecone.from_existing_index(
                        index_name, embeddings)
            else:
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # split text into chunks
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,  # chunk size
                    chunk_overlap=200,  # overlap chunk from previous chunk
                    length_function=len,
                )
                chunks = text_splitter.split_text(text)

                # select embedding
                if ENABLE_HUGGING_FACE:
                    # instructor
                    embeddings = HuggingFaceInstructEmbeddings(
                        model_name="hkunlp/instructor-xl")
                else:
                    embeddings = OpenAIEmbeddings()

                # store the vector as cached file (saved as pickle file)
                if ENABLE_HUGGING_FACE:
                    store_name = f"hf-{pdf.name[:-4]}"
                else:
                    store_name = pdf.name[:-4]

                # check if it exists
                if os.path.exists(f"{store_name}.pk1"):
                    kownledge_base = load_embeddings(store_name)
                else:
                    kownledge_base = store_embeddings(
                        chunks, embeddings, store_name)

            # ask question
            user_question = st.text_input("Ask your question")
            if user_question:
                # find the chunks with similarity (semantic search)
                docs = kownledge_base.similarity_search(
                    query=user_question, k=3)

                # llm integrations:
                # https://python.langchain.com/en/latest/modules/models/llms/integrations.html
                if ENABLE_HUGGING_FACE:
                    llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large",
                                         model_kwargs={"temperature": 0, "max_length": 512})
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    response = chain.run(input_documents=docs,
                                         question=user_question)
                    # show the answer
                    st.write(response)
                else:
                    llm = OpenAI(temperature=0)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs,
                                             question=user_question)
                        # check the price of openAI usage
                        print(cb)
                    # show the answer
                    st.write(response)

    except Exception as e:
        """
        Pinecone: demo account only allow 1 pods by 1 pods
        """
        print("... check exception: ", e)
        st.write(e)


if __name__ == "__main__":
    main()
