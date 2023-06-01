from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
# langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

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
            chunk_size=1000, # chunk size
            chunk_overlap=200, # overlap chunk from previous chunk
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        # FAISS (Facebook AI Similarity Search)
        kownledge_base = FAISS.from_texts(chunks, embeddings)

        # ask question
        user_question = st.text_input("Ask your question")
        if user_question:
            # find the chunks with similarity (semantic search)
            docs = kownledge_base.similarity_search(user_question)
            # llm integrations:
            # https://python.langchain.com/en/latest/modules/models/llms/integrations.html
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=user_question)
              # check the price of openAI usage
              print(cb)
            # show the answer
            st.write(response)


if __name__ == "__main__":
    main()