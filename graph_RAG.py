import os
import streamlit as st

from utilities import configure_graphrag_chain, configure_graphrag_retriever, stream_data
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

COHERE_API_KEY = os.environ["COHERE_API_KEY"]

def main():
    
    st.set_page_config(
        page_title="GraphRAG",
        page_icon="🦜",
        layout="wide",
        initial_sidebar_state="expanded",
        )
    
    graph = Neo4jGraph(
    url="xxx",
    username="xxx",
    password="xxx",
    )

    with st.sidebar:
        
        st.header("Configuration!")
        
        OPENAI_API_KEY = st.text_input(":blue[Enter Your OPENAI API Key:]",
                                    placeholder="Paste your OpenAI API key here (sk-...)",
                                    type="password",
                                    )
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    if OPENAI_API_KEY:
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
            
        st.title("Welcome to GraphRAG!🤖")
        
        pdf = PyPDFLoader("documents\name of your pdf file.pdf")
        loader = pdf.load()
        data = " ".join([doc.page_content for doc in loader])
        
        retriever = configure_graphrag_retriever(data, llm, graph)
        qa_graph_chain = configure_graphrag_chain(retriever, llm)
        
        question = st.text_input("Ask any question about the uploaded file!")
        if st.button("Answer!"):
            with st.spinner("Processing..."):
                response = qa_graph_chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": "session1"}},
                )["answer"]
                st.write_stream(stream_data(response))

if __name__ == "__main__":
    main()
    
