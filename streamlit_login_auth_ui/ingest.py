import logging
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import click
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings
import pdb
import sqlite3
import pandas as pd
from threading import Lock

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

file_cache = os.environ["USER_DIR"]+"/file_list"
csv_files_ingested = os.environ["USER_DIR"]+"/list_of_files_ingested.csv"
PERSIST_DIRECTORY = os.environ["PERSIST_DIRECTORY"]
SOURCE_DIRECTORY = os.environ["SOURCE_DIRECTORY"]
mutex_lock = Lock()
def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")

def checkIfFileAlreadyConsidered(file_path, cache_file):
    mutex_lock.acquire()
    if (os.path.isfile(cache_file)):
        with open(cache_file, 'r') as file:
            print("Trying to load : " +str(file_path))
            load_data = json.load(file)
            print("Load done for: " +str(file_path))
            files = load_data["files"]
            if file_path in files:
                mutex_lock.release()
                return True
    print("File " + str(file_path) + " not present in " + str(cache_file))
    mutex_lock.release()
    return False

def addFileToCache(file_path, cache_file):
    mutex_lock.acquire()
    if (os.path.isfile(cache_file)):
        with open(cache_file, 'r') as file:
            load_data = json.load(file)
            load_data["files"].append(file_path)
    else:
        load_data = {}
        load_data["files"]=[]
        load_data["files"].append(file_path)
    with open(cache_file, 'w') as file:
        json.dump(load_data, file)
    mutex_lock.release()



def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
        if checkIfFileAlreadyConsidered(file_path, file_cache):
            print("Returning null after checkIfFileAlreadyConsidered")
            return None
        print("File path in load_single_document: " +str(file_path))
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path + " loaded.")
            loader = loader_class(file_path)
        else:
            file_log(file_path + " document type is undefined.")
            raise ValueError("Document type is undefined")
        print("Going to add to cache: " + file_path)
        addFileToCache(file_path, file_cache)
        print("Added to cache: " + file_path)
        return loader.load()[0]
    except Exception as ex:
        file_log("%s loading error: \n%s" % (file_path, ex))
        return None


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
            file_log(name + " failed to submit")
            return None
        else:
            data_list = [future.result() for future in futures]
            # return data and file paths
            return (data_list, filepaths)

def preprocess_text(text, in_doc):
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    from string import punctuation
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.text for token in doc]
    punctuation = punctuation + '\n'
    word_list = []
    stop_words = list(STOP_WORDS )
    for word in doc:
        if word.text.lower() not in stop_words:
            if word.text.lower() not in punctuation:
#                if word.text.lower() not in word_list:
                word_list.append(word.text.lower())
    sent=' '.join(map(str, word_list))
    import re
    filteredText=re.sub('[^A-Za-z0-9.@/]+', ' ',sent)
    from nltk.tokenize import TweetTokenizer
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    skills =  tokenizer.tokenize(filteredText)
#Get name
    from pyresparser import ResumeParser
    import warnings
    data = ResumeParser(in_doc.metadata['source']).get_extracted_data()
    name_of_person = data["name"]
    for ent in data["skills"]:
        skills.append(ent)

    final_skills = ""
    for ent in skills:
       final_skills = final_skills + ent + ", "
    final_skills = final_skills.rstrip(", ")
    filteredText = "Resume of candidate " + name_of_person + ":\n" + filteredText
    fp = open(in_doc.metadata['source'], "w")
    fp.write(filteredText)
    fp.close()
    return filteredText,final_skills, name_of_person
def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    print("File Path: " +str(paths))
    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log("Exception: %s" % (ex))
    row=[]
    row.append('Full name')
    row.append('Industry')
    row.append('Job title')
    row.append('Emails')
    row.append('Company Name')
    row.append('Location')
    row.append('Skills')
    row.append('Gender')
    row.append('Linkedin Url')
    row.append('Facebook Url')
    row.append('Twitter Url')
    row.append('Github Url')
    row.append('Github Username')
    row.append('Start Date')
    row.append('Job Summary')
    row.append('Location Country')
    row.append('number of Linkedin Connections')
    row.append('Inferred Salary')
    row.append('Work Experience')
    row.append('Additonal Notes')
    row.append('max git forks')
    row.append('max stars')
    row.append('number of repos')
    row.append('resume')
    final_row=[]
    final_row.append(row)
    
    for doc in docs:
        if doc == None:
            continue
        resume, skills, name = preprocess_text(doc.page_content, doc)
        count = 0
        inner_row = []
        while count < 24:
            if (count == 0):
                inner_row.append(name)
            elif (count == 6):
                inner_row.append(skills)
            elif (count == 23):
                inner_row.append(resume)
            else:
                inner_row.append('')
            count = count + 1
        final_row.append(inner_row)            
        doc.page_content = resume
    if len(final_row) > 1:
        import csv
        with open(os.environ["USER_DIR"] + '/n_data.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(final_row)
        conn = sqlite3.connect(os.environ["USER_DIR"]+'/database1.db')
        c = conn.cursor()
        create_table = '''CREATE TABLE IF NOT EXISTS "userDB"(
"Full name" TEXT, "Industry" TEXT, "Job title" TEXT, "Emails" TEXT,
 "Company Name" TEXT, "Location" TEXT, "Skills" TEXT, "Gender" TEXT,
 "Linkedin Url" TEXT, "Facebook Url" TEXT, "Twitter Url" TEXT, "Github Url" TEXT,
 "Github Username" TEXT, "Start Date" TEXT, "Job Summary" TEXT, "Location Country" TEXT,
 "number of Linkedin Connections" TEXT, "Inferred Salary" TEXT, "Work Experience" INT, "Additonal Notes" TEXT,
 "max git forks" INT, "max stars" INT, "number of repos" INT, "resume" TEXT);'''
        c.execute(create_table)
        file = open(os.environ["USER_DIR"]+'/n_data.csv')
        contents = csv.reader(file)
        insert_records = '''INSERT INTO UserDB ("Full name", "Industry", "Job title", "Emails", "Company Name", "Location", "Skills", "Gender", "Linkedin Url", "Facebook Url", "Twitter Url", "Github Url", "Github Username", "Start Date", "Job Summary", "Location Country", "number of Linkedin Connections", "Inferred Salary", "Work Experience", "Additonal Notes", "max git forks", "max stars", "number of repos", "resume") VALUES(?, ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        c.executemany(insert_records, contents)
        conn.commit()
        conn.close()
                 
    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension == ".py":
                python_docs.append(doc)
            else:
                text_docs.append(doc)
    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within fun_localGPT.py.
    
    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """

    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

#    db = Chroma.from_documents(
#        texts,
#        embeddings,
#        persist_directory=PERSIST_DIRECTORY,
#        client_settings=CHROMA_SETTINGS,
#    )

    fs = LocalFileStore(os.environ["USER_DIR"]+"/store_location")
    store = create_kv_docstore(fs)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=6000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)

    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = ParentDocumentRetriever(
                vectorstore=db,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter)
    retriever.add_documents(texts, ids=None)
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
