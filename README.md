# Graph RAG with LangChain, Neo4j and OpenAI GPT-4

This project showcases an example of a Q&A Conversational Assistant designed to extract valuable information from documents using Langchain, OpenAI GPT-4, the Neo4j graph database, and Streamlit. The system is a Retrieval-Augmented Generation (RAG) model that helps users efficiently retrieve and extract relevant information from their documents through a question-answering interface.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Implementation](#Implementation)

## Introduction
This project demonstrates a Q&A Conversational Assistant that leverages Langchain, OpenAI GPT-4, the Neo4j graph database, and Streamlit to extract meaningful information from documents. It is a Retrieval-Augmented Generation (RAG) system that supports users by efficiently answering questions and retrieving relevant data from their documents.

## Setup
To setup this project on your local machine, follow the below steps:

1. Clone this repository: <code>git clone github.com/EngBaz/Graph-RAG-System</code>

2. Create a virtual enviromnent
   ```console
    $ python -m venv .venv
    $ .venv\Scripts\activate.bat
    ```
3. Install the required dependencies by running <code>pip install -r requirements.txt</code>

4. Create a folder and name it <code>documents</code> to store your PDF files

5. Obtain an API key from OpenAI and Cohere AI. Store the Cohere API key in a <code>.env</code> file with the corresponsding name <code>COHERE_API_KEY</code>.

6. Gather the Noe4j credentials, including <code>url</code>, <code>username</code>, and <code>password</code>
    
7. Note that the project uses OpenAI GPT-4, so you'll need an OpenAI API key. If you prefer to use open-source models from Hugging Face, you can set them up by following these steps:
    ```console
    
    $ pip install langchain huggingface_hub
    $ os.environ['HUGGINGFACE_API_TOKEN'] = 'your_hugging_face_api_token'
    $ llm = HuggingFaceHub(repo_id="model_name", model_kwargs={'temperature': 0.7, 'max_length': 64})
    ```

## Usage

To use the conversational assistant:
1. In the terminal, run the streamlit app: <code> streamlit run graph_RAG.py </code>
2. Write a specific question about your PDF file
3. The assistant will process the input and respond with relevant information


## References

[1] https://arxiv.org/pdf/2408.08921

[2] https://arxiv.org/pdf/2408.04948

[3] https://python.langchain.com/v0.2/docs/introduction/

[4] https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/

[5] https://github.com/ShoaibMajidDar/PDF-chat-bot/blob/main/app.py
