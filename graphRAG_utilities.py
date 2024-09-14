
import time
import streamlit as st
import pandas as pd

from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_experimental.graph_transformers import LLMGraphTransformer


def configure_graphrag_retriever(data, llm, graph):
    """
    Configures a hybrid search-enabled question-answer (QA) chain using vector search and keyword search.
    The function processes raw data, converts it into graph documents, stores them in a graph database, 
    and sets up a retriever that integrates both search methods. The results are reranked for improved 
    context selection.

    Args:
        data: Raw input data to be processed, chunked, and added to the graph database.
        llm: A language model (LLM) for generating graph documents and for the QA chain.
        graph: The Neo4j graph database instance where the transformed documents are stored.

    Returns:
        qa_graph_chain: A QA chain with hybrid search and context reranking for question-answering.
    """
    semantic_chunker = SemanticChunker(OpenAIEmbeddings(), 
                                       breakpoint_threshold_type="percentile",
                                       )
    chunked_documents = semantic_chunker.create_documents([data])
    
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(chunked_documents)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True,
        )
    
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        url="xxx",
        username="xxx",
        password="xxx",
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
        )
    vector_retriever = vector_index.as_retriever(search_kwargs={"k": 5})
    
    keyword_retriever = BM25Retriever.from_documents(chunked_documents)
    keyword_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], 
                                           weights=[0.5, 0.5],
                                           )
    
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever,
        )
    
    return compression_retriever



def configure_graphrag_chain(retriever, llm):
    """
    Configures a Retrieval-Augmented Generation (RAG) chain that uses a retriever and a large language model
    to handle conversational question answering with session-specific history.

    Args:
        retriever: The document retriever responsible for fetching relevant context from a knowledge base.
        llm: The large language model used to generate answers to the user's questions.

    Returns:
        conversational_rag_chain: A RAG chain that handles input messages, retrieves relevant context,
        manages chat history, and generates responses based on both the retrieved context and prior conversations.
    """
    contextualize_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ]
        )
        
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        contextualize_prompt,
        )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer concise and clear.\
    {context}"""
        
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ]
        )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, 
                                       question_answer_chain,
                                       )

    store = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
            return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        )

    return conversational_rag_chain


def stream_data(response):
    
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)
