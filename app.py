from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import os
import pickle
import pinecone
# langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# initialize pinecone
# reference: https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/Chat_with_Any_Documents_Own_ChatGPT_with_LangChain.ipynb
PINECONE_API_KEY = ""
PINECONE_ENV = ""
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)
index_name = "langchain-demo"


def main():
    load_dotenv()

    # streamlit
    st.set_page_config(page_title="Ask your pdf")
    st.header("Ask your pdf")
    pdf = st.file_uploader("Upload your pdf here..", type="pdf")

    # read pdf
    if pdf is not None:
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

        # store the vector as cached file (saved as pickle file)
        # store_name = pdf.name[:-4]
        # if os.path.exists(f"{store_name}.pk1"):
        #     with open(f"{store_name}.pk1", "rb") as f:
        #         kownledge_base = pickle.load(f)
        #     print("loaded from cached")
        # else:
        #     # create embeddings
        #     embeddings = OpenAIEmbeddings()
        #     # FAISS (Facebook AI Similarity Search)
        #     kownledge_base = FAISS.from_texts(chunks, embeddings)
        #     # save it as cached
        #     with open(f"{store_name}.pk1", "wb") as f:
        #         pickle.dump(kownledge_base, f)

        index_name = "langchain-demo"
        embeddings = OpenAIEmbeddings()
        kownledge_base = Pinecone.from_existing_index(index_name, embeddings)

        # ask question
        user_question = st.text_input("Ask your question")
        if user_question:
            # find the chunks with similarity (semantic search)
            docs = kownledge_base.similarity_search(query=user_question, k=3)
            # llm integrations:
            # https://python.langchain.com/en/latest/modules/models/llms/integrations.html
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=user_question)
                # check the price of openAI usage
                print(cb)
            # show the answer
            st.write(response)


if __name__ == "__main__":
    main()
