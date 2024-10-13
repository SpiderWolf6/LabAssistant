# import necessary libraries
import os
import io
import sys
import time
import shutil
import logging
import chromadb
import chainlit as cl

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_community.document_loaders import UnstructuredExcelLoader
from chromadb.config import Settings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import PyPDF2

# store azure configuration in variables
OPENAI_API_TYPE = "Azure"
OPENAI_API_VERSION = "2024-06-01"
OPENAI_API_BASE = "https://sm-chatbot.openai.azure.com/"
OPENAI_API_KEY = "9afbddb3414f48358545acc11404fd9d"
DATA_PATH = "C:/Users/soham/Work/MSKCC/summer_2024/gen_ai/data/papers/"
CHROMA_PATH = "chroma"
ai_model = "gpt-35-turbo"
embeddings_model = "text-embedding-ada-002"

# set environment variables
os.environ['OPENAI_API_TYPE'] = OPENAI_API_TYPE
os.environ['OPENAI_API_VERSION'] = OPENAI_API_VERSION
os.environ['azure_endpoint'] = OPENAI_API_BASE
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# set variables for model (used later)
temperature = 0.9
max_size_mb = 100
max_files = 10
text_splitter_chunk_size = 1000
text_splitter_chunk_overlap = 10
embeddings_chunk_size = 16
max_retries = 5
retry_min_seconds = 1
retry_max_seconds = 5
timeout = 30

# configure system prompt
system_template = """Use the following pieces of context to answer questions.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

You will be the lab research assistant to the Benjamin Greenbaum Lab at MSKCC.
You will learn about the labs research interests from the online profile: https://www.mskcc.org/research-areas/labs/benjamin-greenbaum/
You will be pre-trained with research papers of interest to the Greenbaum Lab. You will also read any reference papers. 

You will be speaking to researchers from the Greenbaum Lab. Start by introducing yourself and informing how you may help.
Answer any questions to support the lab's research work, with context and sources in mind.

ALWAYS return a "SOURCES" and "RELATED WORKS" part in your answer.
The "SOURCES" part should be a reference to the research paper from which you got your answer, provide the link.
The "RELATED WORKS" part should provide links to suggested related literature from Nature or PubMed.

Example of your response should be:

```
The answer is foo
SOURCES: links
RELATED WORKS: links
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

# configure a logger
logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# on chainlit run, implement following
@cl.on_chat_start
async def start():
    print("started")

    # create empty list for storing context documents and id
    documents = []
    ids = []

    # create Azure embeddings object (this example uses text-embedding-ada-002
    embeddings = AzureOpenAIEmbeddings(
        model=embeddings_model,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_type=OPENAI_API_TYPE,
        openai_api_key=OPENAI_API_KEY,
        azure_endpoint=OPENAI_API_BASE,
        azure_deployment=embeddings_model,
        max_retries=max_retries,
        retry_min_seconds=retry_min_seconds,
        retry_max_seconds=retry_max_seconds,
        chunk_size=embeddings_chunk_size,
        timeout=timeout,
    )

    # Authenticate and initialize PyDrive
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Authenticate
    drive = GoogleDrive(gauth)

    # commented out, but use the two lines below to remove existing chroma if needed
    # if os.path.exists(CHROMA_PATH):
    #     shutil.rmtree(CHROMA_PATH)

    # if chroma path exists, load existing document ids

    if os.path.exists(CHROMA_PATH):
        logger.info("existing Chroma db found")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        ids = db.get()["ids"]
        logger.info(ids)
        time.sleep(5)

    # if 0 ids, then there is no existing chroma db, so iterate through files to create new chromadb
    if len(ids) == 0:
        print("empty_ids")

        # Download files from a specific folder
        folder_id = '1RiuJcPnhA81rcNiJInnHE_FmoxcDJMkT'  # Replace with the actual folder ID
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and mimeType='application/pdf'"}).GetList()

        # Iterate through and process each PDF file
        for file in file_list:
            print(f'Downloading {file["title"]}')
            file.GetContentFile(file['title'])

            # Open and read the PDF file
            with open(file['title'], 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    text = reader.pages[page_num].extract_text()
                    documents += [Document(
                        page_content=text,
                        metadata={"source": "local"},
                        id=file['title'] + " section " + str(page_num)
                    )]
                    ids += [file['title'] + " section " + str(page_num)]

            db = Chroma.from_documents(
                documents, embeddings, persist_directory=CHROMA_PATH, ids=ids
            )
    # if ids exist, search for unique (new) ids and add only those documents to existing chromadb
    else:
        new_documents = []
        new_ids = []
        # Download files from a specific folder
        folder_id = '1RiuJcPnhA81rcNiJInnHE_FmoxcDJMkT'  # Replace with the actual folder ID
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and mimeType='application/pdf'"}).GetList()

        # Iterate through and process each PDF file
        for file in file_list:
            if not any(file["title"] in i for i in ids):
                print(f'Downloading {file["title"]}')
                file.GetContentFile(file['title'])

                # Open and read the PDF file
                with open(file['title'], 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(reader.pages)
                    for page_num in range(num_pages):
                        if id not in ids:
                            text = reader.pages[page_num].extract_text()
                            new_documents += [Document(
                                page_content=text,
                                metadata={"source": "local"},
                                id=file['title'] + " section " + str(page_num)
                            )]
                            new_ids += [file['title'] + " section " + str(page_num)]

        if (len(new_ids) != 0):
            Chroma.add_documents(documents=new_documents, embeddings=embeddings, ids=new_ids, persist_directory=CHROMA_PATH, self=db)

    # empty all_texts and metadatas tracks message history between User and chatbot
    all_texts = []
    metadatas = [{"source": f"{i}-pl"} for i in range(len(all_texts))]

    # create OpenAI object (this example uses gpt-3.5-turbo)
    llm = AzureChatOpenAI(
        model=ai_model,
        openai_api_type=OPENAI_API_TYPE,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=OPENAI_API_KEY,
        azure_endpoint=OPENAI_API_BASE,
        temperature=temperature,
        azure_deployment=ai_model,
        streaming=True,
        max_retries=max_retries,
        timeout=timeout,
    )

    # create a chain that uses the Chroma vector store and llm
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    # save the metadata and texts in the user sessions
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", all_texts)

    # send the first message from the chatbot, indicating it is ready for conversation
    content = "Hello" + ", I am the research assistant to the Greenbaum Lab. I would be happy to help with any questions related to the labs research interests, new paper ideas, statistical models, or anywhere else I could provide input."
    logger.info(content)
    msg = cl.Message(content=content, author="Chatbot")
    await msg.send()

    # store the chain in the user session
    cl.user_session.set("chain", chain)


# when User sends message, implement following
@cl.on_message
async def main(message: cl.Message):
    # retrieve the chain from the user session
    chain = cl.user_session.get("chain")

    # create a callback handler
    cb = cl.AsyncLangchainCallbackHandler()

    # get the response from the chain
    response = await chain.acall(message.content, callbacks=[cb])
    logger.info("Question: [%s]", message.content)

    # get the answer and sources from the response
    answer = response["answer"]
    sources = response["sources"].strip()
    source_elements = []

    # if debug:
    #     logger.info("Answer: [%s]", answer)

    # get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()