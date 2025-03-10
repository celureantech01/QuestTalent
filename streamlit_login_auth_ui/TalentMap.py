import pdb
import torch
import re
import random
import os
import logging
from numba import cuda
import subprocess
import streamlit as st
import csv
import json
import utils
from langchain.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from prompt_template_utils import get_prompt_template
from streamlit_lottie import st_lottie_spinner
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import pdb
from langchain.schema.retriever import BaseRetriever
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.document import Document
from typing import List
from typing import Dict
from typing import Any
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.vectorstores import VectorStore
import logging.handlers
import threading
import pickle
import pdb
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import TokenTextSplitter
from langchain.storage import LocalFileStore
from langchain.retrievers import BM25Retriever, EnsembleRetriever


cache_file="./linkedin_data"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        if 'query_start' in st.session_state:
            self.text = ""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token 
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

def display_chat_history():
    """
    Displays the chat history.
    """
    count = 0
    for ent in st.session_state['responses']:
        if count < len(st.session_state['requests']):
            with st.chat_message('user'):
                st.write(st.session_state['requests'][count])
        with st.chat_message('assistant'):
            st.write(ent)

        count = count + 1


def ret_linkedin_data_from_cache(linkedin_profile_url: str):
    if (os.path.isfile(cache_file)):
        with open(cache_file, 'rb') as file:
            load_data = pickle.load(file)
            if linkedin_profile_url in load_data.keys():
                return load_data[linkedin_profile_url]

    return {}


def add_linkedin_data_to_cache(link_data: dict):
    if (os.path.isfile(cache_file)):
        with open(cache_file, 'rb') as file:
            load_data = pickle.load(file)
            new_dict = load_data | link_data
    else:
        new_dict = link_data

    with open(cache_file, 'wb') as file:
        pickle.dump(new_dict, file)


def scrape_linkedin_profile(linkedin_profile_url: str):
#    """
#    Getting Linkedin data using nubela's proxy curl api
#    """

    user_dict = ret_linkedin_data_from_cache(linkedin_profile_url)
    if user_dict != {}:
        return user_dict

    user_dict = {}    
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"

    job_search_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin/company/job"

    response = requests.get(
                api_endpoint,
                params = {"url": "https://www."+linkedin_profile_url},
                headers = {"Authorization": f"Bearer {os.getenv('PROXY_CURL_API')}"}
                )

    data = response.json()
    include_keys = ['education',
                    'articles',
                    'certifications',
                    'inferred_salary',
                    'interests',
                    'accomplishment_patents',
                    'country',
                    'city',
                    'state',
                    'accomplishment_organisations',
                    'accomplishment_publications',
                    'accomplishment_honors_awards',
                    'accomplishment_projects',
                    'accomplishment_test_scores',
                    'volunteer_work',
                    'accomplishment_courses',
                    'industry',
                    'personal_numbers',
                    'gender',
                    'follower_count',
                    'occupation',
                    'experiences',
                   ]

    data = {
            key: value
            for key, value in data.items()
            if key in include_keys and value not in ([], "", None,'')
            }

    additional_text = ""

    for ent in  include_keys:
        if ent in data.keys():
            additional_text = additional_text + "\n"+ str(ent) + " details. "
            count = 1
            if isinstance(data[ent], list):
                additional_text = additional_text + " There are " + str(len(data[ent])) + " " + str(ent) + " entries:"
                for entry in data[ent]:
                    if isinstance(entry, str):
                        additional_text = additional_text + " " + str(entry) + "\n"
                    elif isinstance(entry, dict):
                        additional_text = additional_text + "\nEntry "+ str(count) + ": "
                        count = count + 1
                        for key in entry.keys():
                            if entry[key] in ([], "", None,''):
                                continue
                            additional_text = additional_text + str(key) + ": " + str(entry[key]) + "\n"
            else:
                additional_text = additional_text + " " + str(data[ent]) + "\n"

    user_dict[linkedin_profile_url] = {"additional_info" : additional_text}

    add_linkedin_data_to_cache(user_dict)
    return user_dict[linkedin_profile_url]

class thread(threading.Thread): 
    def __init__(self, thread_name, thread_ID): 
        threading.Thread.__init__(self) 
        self.thread_name = thread_name 
        self.thread_ID = thread_ID 
 
        # helper function to execute the threads
    def run(self): 
        print("Entered Linkedin data gather thread\n")
        linkDict = {}
        while True:
            if ('linkedinList' not in st.session_state):
                sleep(2)
            print("Got Linkedin list : " + str(st.session_state.linkedinList))
            break

        for ent in st.session_state.linkedinList:
            linkdata = scrape_linkedin_profile(ent)
            linkDict[ent] = linkdata
        st.session_state.linkDict = linkDict


def setLoggingHandler():
    handler = logging.handlers.WatchedFileHandler(
            os.environ.get("LOGFILE", "/var/log/TalentMap.log"))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
#    root.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
    root.addHandler(handler)

def getFollowUpState():
    if 'FOLLOW_UP' in st.session_state:
        followup_reload = st.session_state.FOLLOW_UP
    else:
        followup_reload = ''

    return followup_reload

def reload():
    if 'ANS' in st.session_state:
        del st.session_state['ANS'] 
    if 'disabled' in st.session_state:
        del st.session_state.disabled
    if 'SQL_LLM' in st.session_state:
        del st.session_state.SQL_LLM
    if 'EMBEDDINGS' in st.session_state:
        del st.session_state.EMBEDDINGS
    if 'DB' in st.session_state:
        del st.session_state.DB
    if 'RETRIEVER' in st.session_state:
        del st.session_state.RETRIEVER
    if 'CUSTOM_RETRIEVER' in st.session_state:
        del st.session_state.CUSTOM_RETRIEVER
    if 'LLM' in st.session_state:
        del st.session_state["LLM"]
    if "SQL_LLM" in st.session_state:
        del st.session_state["SQL_LLM"]
    if "SESSION_START" in st.session_state:
        del st.session_state["SESSION_START"]
    if "QAStruct" in st.session_state:
        del st.session_state.QAStruct
    if 'resume' in st.session_state:
        del st.session_state['resume']
    if 'INGESTION' in st.session_state:
        del st.session_state.INGESTION
    if 'FOLLOW_UP' in st.session_state:
        del st.session_state.FOLLOW_UP
    return

def disable():
    st.session_state.disabled = True

def enable():
    st.session_state.disabled = False

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. Answers should be specific to Question asked. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer. Consider resume and profile as same.\

    {context} 

    {history}
    Question: {question}
    Helpful Answer:"""
    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferWindowMemory(input_key="question", memory_key="history", k=10)

    return prompt,memory

def convert_json_csv():
    json_file = os.environ["USER_DIR"]+"/file_list"
    csv_data = []
    if os.path.isfile(json_file):
        with open(json_file) as json_file:
            data = json.load(json_file)
        file_names = data['files']
        for fil in file_names:
            baseName = os.path.basename(fil)
            csv_data.append(baseName)
    dictn = {'FileName': csv_data}
    df = pd.DataFrame(dictn) 
    dataRet = df.to_csv(index=False).encode('utf-8')
    return dataRet

def disable_prompt():
   st.session_state.download = 'download'
        
def setUpSideBar():
    chat_container = st.empty()
    with st.sidebar:
        col1, col2 = st.columns([1,1])
        with col1:
            csv_data = convert_json_csv()
            st.download_button(
                label="files ingested",
                data=csv_data,
                on_click = disable_prompt,
                file_name=os.environ["USER_DIR"] + "/list_of_files_ingested.csv",
                mime='text/csv',
            )

        if os.path.isfile(os.environ["USER_DIR"] + "/local_chat_history/qa_log.csv"): 
            with col2:
                df = pd.read_csv(os.environ["USER_DIR"] + "/local_chat_history/qa_log.csv")
                csv_data = df.to_csv(index=False).encode('utf-8')                
                st.download_button(
                    label="chat snippet",
                    data=csv_data,
                    on_click = disable_prompt,
                    file_name=os.environ["USER_DIR"] + "/local_chat_history/qa_log.csv",
                    mime='text/csv',
                )
        if 'download' in st.session_state:
            del st.session_state.download
#        st.button("Refresh", type="secondary", on_click=reload)
        st.title("TalentMap")
        st.image('TalentMap.jpeg', caption='Making talent search easier', width=200 )
        add_vertical_space(2)
        dir_name = os.path.join(st.session_state.userDir+"/SOURCE_DOCUMENTS")
        isExist = os.path.exists(dir_name)
        if not isExist:
            os.makedirs(dir_name)
        if 'upload_key' not in st.session_state:
            st.session_state.upload_key = random.random()

        uploaded_files = st.file_uploader("Ingest files", accept_multiple_files=True, key=st.session_state.upload_key, type=None)
        orig_upload_len = 0
        if uploaded_files:
            if 'FILES_UPLOADED' not in st.session_state:
                st.session_state['FILES_UPLOADED'] = []
                orig_upload_len = 0
            else:
                orig_upload_len = len(st.session_state['FILES_UPLOADED'])
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state['FILES_UPLOADED']:
                    continue
                else:
                    st.session_state['FILES_UPLOADED'].append(uploaded_file.name)
                st.session_state.INGESTION = "INGESTION"
                file_path = os.path.join(st.session_state.userDir+"/SOURCE_DOCUMENTS", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                if file_path.endswith((".tar", ".zip")):
                    if file_path.endswith(".tar"):
                        cwd = os.getcwd()
                        os.chdir(st.session_state.userDir+"/SOURCE_DOCUMENTS")
                        os.system("tar -xvf " + uploaded_file.name)
                        os.chdir(cwd)
                    elif file_path.endswith(".zip"):
                        cwd = os.getcwd()
                        os.chdir(st.session_state.userDir+"/SOURCE_DOCUMENTS")
                        os.system("unzip " + uploaded_file.name)
                        os.chdir(cwd)
        with st.spinner("Please wait ingestion in progress ....."):
            if 'FILES_UPLOADED' in st.session_state and orig_upload_len != len(st.session_state['FILES_UPLOADED']):
                os.environ["PERSIST_DIRECTORY"] = st.session_state.userDir + "/DB"
                os.environ["SOURCE_DIRECTORY"] = st.session_state.userDir+"/SOURCE_DOCUMENTS"
                os.environ["USER_DIR"] = st.session_state.userDir
                lurl = "https://lottie.host/ce4f03ca-b936-4ec3-a238-784a070dd7f1/I7DSZEmuV7.json"
                ltitle = load_lottieurl(lurl)
                with chat_container:
                    with st.chat_message('assistant'):
                        st.write("**Please wait ..this pane will be active once resume ingestion completes**")
                        with st_lottie_spinner(ltitle, key="progress", width =300, loop=True):
                            os.system("python ingest.py")
                chat_container.empty()
                if 'INGESTION' in st.session_state:
                    del st.session_state.INGESTION
                if 'FOLLOW_UP' in st.session_state:
                    del st.session_state.FOLLOW_UP
                if 'resume' in st.session_state:
                    del st.session_state.resume

def getSkillBasedOnRole(llm, query):
    out = llm.predict("provide only skill as single word without any special characters like space, \n, quotes etc, required for below job description, in python list format for example role_skill=[skill1, skill2]. Show only role_skill list. Consider only context. context: " + query)

    final_skills = []
    try:
        startskill = out.split("[")[1]
        skills = startskill.split("]")[0]
        skill_sp = skills.split(",")
        for ent in skill_sp:
            nent = re.search(r"\w+ *\w*", ent)
            ent = nent.group()
            ent = ent.lstrip("\"")
            ent = ent.rstrip("\"")
            final_skills.append(ent)
    except:

        pass

    return final_skills


def getWorkExperience(llm, query):

    return


class FollowUpCustomRetriever(BaseRetriever):
    sQuery: dict
#    @cuda.jit
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
                       chunk_size=100000,
                        chunk_overlap=500,
                        length_function=len,
                        is_separator_regex=False,
                        )
        resumeList = []
        for key in sQuery.keys():
            resumeList.append(sQuery[key])
        texts = text_splitter.create_documents(resumeList)
        return texts

class CustomVecRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        if 'resume' in st.session_state:
            if 'immediate_follow' in st.session_state:
                if 'docs' in  st.session_state:
                    return st.session_state.docs

            logging.debug("Inside Follow up for vector. Num of docs: " + str(len(st.session_state.resume)))
            text_splitter = RecursiveCharacterTextSplitter(
                           chunk_size=100000,
#                           chunk_size=3000,
                            chunk_overlap=100,
                            length_function=len,
                            is_separator_regex=False,
                            )
            resumeList = []
            for key in st.session_state.resume.keys():
                resumeList.append(str(st.session_state.resume[key]))
                print("\n Follow up: " + str(key))
            texts = text_splitter.create_documents(resumeList)
            logging.debug('Follow up documents: ' + str(texts))
            return texts
        else:
#            documents = self.vectorstore.get_relevant_documents(query, callbacks=run_manager.get_child())
            documents = st.session_state.retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
            count = min(5, len(documents))
            return documents[:count]


class CustomRetriever(BaseRetriever):
    sQuery: dict = {}
#    @cuda.jit
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if 'resume' in st.session_state:
            if 'immediate_follow' in st.session_state:
                if 'docs' in  st.session_state:
                    return st.session_state.docs
            logging.debug("Inside Follow up for first time SQL. Num of docs: " + str(len(st.session_state.resume)))
#            return st.session_state.extra_doc
            text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=100000,
                            chunk_overlap=100,
                            length_function=len,
                            is_separator_regex=False,
                            )
            resumeList = []
            for key in st.session_state.resume.keys():
                resumeList.append(str(st.session_state.resume[key]))
                print("\n Follow up: " + str(key))
            texts = text_splitter.create_documents(resumeList)
            logging.debug('Follow up documents: ' + str(texts))
            return texts
        if os.path.isfile(st.session_state.username+"/database1.db"):
            db = SQLDatabase.from_uri("sqlite:///" + st.session_state.username+"/database1.db", max_string_length=6000)
        else:
            db = SQLDatabase.from_uri("sqlite:///" + "/database1.db", max_string_length=6000)
        if 'SQL_LLM' not in st.session_state:
            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
            st.session_state.SQL_LLM = llm

        skillList = getSkillBasedOnRole(st.session_state['SQL_LLM'], query)
        skills = self.sQuery['skills']
        workexp = self.sQuery['workEx']
        skill_text = ''
        try:
            query = "SELECT \"Linkedin Url\",\"resume\" FROM \"userDB\" WHERE "
            order_by_text = '\n ORDER BY ('
            for sk in skills:
                skill_text = skill_text + sk +  ' '
                sk = sk.lstrip(" ")
                sk = sk.rstrip(" ")
                sk = sk.lstrip("\'")
                sk = sk.rstrip("\'")
                sk = sk.lstrip(" ")
                sk = sk.rstrip(" ")
                query = query + " LOWER(\"Skills\") LIKE '%" + sk.lower() + "%' OR " 
                order_by_text = order_by_text + "CASE WHEN LOWER(\"Skills\") LIKE '%" + sk.lower() + "%' THEN 1 ELSE 0 END +"

            for sk in skillList:
                skill_text = skill_text + sk +  ' '
                sk = sk.lstrip(" ")
                sk = sk.rstrip(" ")
                sk = sk.lstrip("\'")
                sk = sk.rstrip("\'")
                sk = sk.lstrip(" ")
                sk = sk.rstrip(" ")
                query = query + " LOWER(\"Skills\") LIKE '%" + sk.lower() + "%' OR "
                order_by_text = order_by_text + "CASE WHEN LOWER(\"Skills\") LIKE '%" + sk.lower() + "%' THEN 1 ELSE 0 END +"

            query = query.rstrip("OR ")

            if workexp != '':
                query = query + " AND \"Work Experience\" > " + str(int(workexp))
            order_by_text = order_by_text.rstrip("+")
            query = query + order_by_text + " ) DESC LIMIT 15;"
            
        except:
            pass
        if skill_text == '' and workexp == '':
            text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=100000,
                        chunk_overlap=500,
                        length_function=len,
                        is_separator_regex=False,
                    )

            texts = text_splitter.create_documents([])
            return texts
        print("Query: " + str(query))
        logging.debug('Query: ' + str(query))
        dbOut=db.run(query)
        try:
            linkDbList= [(s[0], s[1]) for s in eval(dbOut)]
#            st.session_state.linkedinList = linkedinList
        except:
            dbList = []
            linkDbList = []
            pass
        dbList = []
        for ent in linkDbList:
            linkdinURL = ent[0]
            dbList.append(ent[1])
#            try:
#                data = scrape_linkedin_profile(linkdinURL)
#            except:
#                pass
#
        print("Total results from SQL : " + str(len(dbList)))
        print("Try to get some documents from vector store also\n")
        skills_text = 'search candidates based on maximum matching skills. '
        for sk in skillList:
            skills_text = skills_text + str(sk) + ", "
        for sk in skills:
            skills_text = skills_text + str(sk) + ", "

        print("Going to search for following query in Vec DB. " + str(skills_text))
        vecList = []
        if 'RETRIEVER' in st.session_state:
            vecDoc = st.session_state.RETRIEVER.get_relevant_documents(skills_text, callbacks=run_manager.get_child())
            for i in vecDoc:
                vecList.append(i.page_content)
        dbList.extend(vecList)
        text_splitter = RecursiveCharacterTextSplitter(
#                       chunk_size=3000,
                       chunk_size=100000,
                        chunk_overlap=100,
                        length_function=len,
                        is_separator_regex=False,
                        )
        texts = text_splitter.create_documents(dbList)
        st.session_state.docs = texts
        return texts

def login():
    __login__obj = __login__(auth_token = "pk_prod_N8PZ9A6BY446D0NSY3XAFVVJ26TA",
                        company_name = "TalentMaps",
                        width = 200, height = 250,
                        logout_button_name = 'Logout', hide_menu_bool = False,
                        hide_footer_bool = False,
                        lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

    LOGGED_IN = __login__obj.build_login_ui()
    if LOGGED_IN == True:
        username= __login__obj.get_username()
        st.session_state.userDir = os.getcwd() + "/" + str(username)
        st.session_state.username = username   
        os.environ["PERSIST_DIRECTORY"] = st.session_state.userDir + "/DB"
        if not os.path.isfile(os.environ["PERSIST_DIRECTORY"]) or not os.listdir(os.environ["PERSIST_DIRECTORY"]):
            os.environ["PERSIST_DIRECTORY"] = os.getcwd() + "/DB"
            
        os.environ["SOURCE_DIRECTORY"] = st.session_state.userDir+"/SOURCE_DOCUMENTS"
        os.environ["USER_DIR"] = st.session_state.userDir

    return LOGGED_IN

def getRetriever(userDir, embeddings):
    db = Chroma(persist_directory=os.environ['PERSIST_DIRECTORY'], embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    
    fs = LocalFileStore(userDir+"/store_location")
    parent_store = create_kv_docstore(fs)
    parent_splitter = TokenTextSplitter(chunk_size=6000, chunk_overlap=0)
    child_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=0)
    
    parent_retriever = ParentDocumentRetriever(
                vectorstore=db,
                docstore=parent_store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter)
    ensemble_retriever = EnsembleRetriever(retrievers=[parent_retriever], weights=[1])

    return ensemble_retriever

def setUpSessionState():
    if torch.backends.mps.is_available():
        DEVICE_TYPE = "mps"
    elif torch.cuda.is_available():
        DEVICE_TYPE = "cuda"
    else:
        DEVICE_TYPE = "cpu"
    DEVICE_TYPE = "cpu"

    if "EMBEDDINGS" not in st.session_state:
        EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
        st.session_state.EMBEDDINGS = EMBEDDINGS

    if "DB" not in st.session_state:
        os.environ["PERSIST_DIRECTORY"] = st.session_state.userDir + "/DB"
        DB = Chroma(
                persist_directory=os.environ['PERSIST_DIRECTORY'],
                embedding_function=st.session_state["EMBEDDINGS"],
                client_settings=CHROMA_SETTINGS,
                )
        st.session_state.DB = DB

    if 'RETRIEVER' not in st.session_state:
        currentRetreiver = getRetriever(os.environ['USER_DIR'], st.session_state.EMBEDDINGS)
        st.session_state.retriever = currentRetreiver
        RETRIEVER = CustomVecRetriever(vectorstore=DB.as_retriever(search_type="mmr", search_kwargs={"k": 3}))
        st.session_state.RETRIEVER = RETRIEVER

    if 'CUSTOM_RETRIEVER' not in st.session_state:
        CUSTOM_RETRIEVER = CustomRetriever()
        st.session_state.CUSTOM_RETRIEVER = CUSTOM_RETRIEVER
    if "LLM" not in st.session_state:
    #    LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
#    query = "show candidates suitable for role of front end developer"

#    result = LLM("provide only skill as single word, required for below job description, in python list format for example role_skill=[skill1, skill2]. Show only role_skill list. " + query)
        chat_box = st.empty()
        st.session_state.stream_handler = StreamHandler("", display_method='write')
        SQL_LLM = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
        LLM = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, streaming= True, callbacks=[st.session_state.stream_handler])
        st.session_state["LLM"] = LLM
        st.session_state["SQL_LLM"] = SQL_LLM

def getUniqueSkills():
    df = pd.read_csv("./9_02.csv")
    skillList = list(df['Skills'].unique())
    uniqeskillList = []
    for ent in skillList:
        try:
            nl = ent.split(",")
            for ent1 in nl:
                if ent1 in uniqeskillList:
                    pass
                else:
                    uniqeskillList.append(ent1)
        except:
            continue
    return uniqeskillList

	
def setUpRetrieverQAPipeline(Skills, workEx):
    if 'stream_handler' in st.session_state:
        st.session_state.stream_handler.container = st.empty()
    prompt, memory = model_memory()
    skills_work = {'skills': Skills,
                   'workEx': workEx}

    QAStruct = RetrievalQA.from_chain_type(
                      st.session_state["LLM"],
#                          chain_type="map_reduce",
                      chain_type="stuff",
                      retriever= CustomRetriever(sQuery=skills_work),
                      return_source_documents=True,
                      verbose=True,
                      chain_type_kwargs={"prompt": prompt, "memory": memory},
#                         chain_type_kwargs={"question_prompt": prompt, "memory": memory},
                      )
    st.session_state.QAStruct = QAStruct



def inference_start():
    uniqeskillList = getUniqueSkills()
    Skills = st.multiselect(label = "Choose a skill", options = uniqeskillList)
    workEx = st.text_input("Input work experience greater than", on_change=enable)
    Prompt = st.text_area("Input Job Description here", on_change=enable)
    if 'disabled' not in st.session_state:
        st.session_state.disabled = False
    submit=st.button("Submit Query", on_click=disable, disabled=st.session_state.disabled)
    st.session_state.START_SESSION = True
    setUpRetrieverQAPipeline(Skills, workEx) 
    final_answer = ""
    if submit:
        with st.spinner("Please wait... (results will be streaming so you might need to scroll down the page)"):
            if 'stream_handler' in st.session_state:
                st.session_state.stream_handler.container = st.empty()
            response = st.session_state['QAStruct']("Find all candidates suitable for job mentioned next. " + Prompt + ". for each selected candidate explain in not more than 150 words why you have selected that candidate. Also show their linkedin url with name as a hyperlink. Also Show their git statistics.\n")
            answer, docs = response["result"], response["source_documents"]
            final_answer = answer
                       
        if 'resume' not in st.session_state:
            resume = {}
            for docent in docs:
                name = re.findall(r"Resume of candidate \w+ *\w+", docent.page_content)
                if name == []:
                    try:
                        from pyresparser import ResumeParser
                        fp = open("./tmpfile", "w")
                        fp.write(docent.page_content)
                        fp.close()
                        data = ResumeParser("./tmpfile").get_extracted_data()
                        name = data["name"]
                    except:
                        name = []
                        pass
                if name == []:
                    continue                
                name = name[0].split('Resume of candidate ')[1]
                if name not in resume.keys():
                    resume[name] = docent.page_content
                    obj = re.search(r"linkedin.com\/\w*\/\w*", resume[name])
                    if obj != None:
                        linkedin_url = obj.group()
                        linkedin_url = linkedin_url.lstrip("https://www.")
                        userDict = scrape_linkedin_profile(linkedin_url)
                        for keys in userDict.keys():
                            resume[name] = resume[name] + "\n" + str(keys) + "\n" + str(userDict[keys])
                            print("Added linkedin data to : " + name +"\n" + str(userDict[keys]))
            st.session_state['resume'] = resume
            print("Total resumes : " + str(len(st.session_state['resume'])))
        count = 0
        while count < 3:
            st.session_state.immediate_follow = "immediate_follow"
            response = st.session_state['QAStruct']("show me new candidates suitable for job apart from which you have already shown. Make sure you are not showing candidates which you have already selected, also continue the count from where it was left. If there are no new candidate say Above all are selected candidates.")
            answer, docs = response["result"], response["source_documents"]
            obj = re.search(r"Above all are selected candidates", answer)
            if obj != None:
                break
            final_answer = final_answer + "\n"+ answer
            count = count + 1
        if 'immediate_follow' in st.session_state:
            del st.session_state.immediate_follow
        if 'docs' in st.session_state:
            del st.session_state.docs
        if 'stream_handler' in st.session_state:
            st.session_state.stream_handler.container.empty()
            st.session_state.stream_handler.text = ""
        st.subheader("👉 " + "TalentMap Recommendations: ")
        st.write(final_answer)
        utils.log_to_csv(Prompt, final_answer)
        if 'ANS' not in st.session_state:
            st.session_state['responses']=[]
            st.session_state['requests']=[]
            if final_answer != '':
                st.session_state['ANS'] = final_answer
                st.session_state['responses'].append(final_answer)
                st.session_state['requests'].append(Prompt)
                st.session_state.FOLLOW_UP = "Follow up"
        st.success("Hope I was able to save your time")
        st.button("Continue chatting...", type="secondary")
def set_query_start():
    st.session_state.query_start = "query_start"

def inference_followup():
#    if "query_start" not in st.session_state:
    chat_container = st.container()
    prompt_container = st.container()


    if "prompt_key" not in st.session_state:
        st.session_state.prompt_key = random.random()
    
    with prompt_container:
        if 'stream_handler' in st.session_state:
            st.session_state.stream_handler.container = st.empty()

        dis = False
        if 'download' in st.session_state:
            dis = True            
        query = st.text_input('Prompt: ', placeholder='Enter your prompt here..', disabled = dis, key=st.session_state.prompt_key)
        answer = ''
        if query:
            st.session_state.query_start = "query_start"
            with st.spinner('Generating Response...(respose will be streaming so you might need to scroll down the page)'):
                response = st.session_state['QAStruct'](query)
                answer, docs = response["result"], response["source_documents"]
            st.session_state.requests.append(query)
            st.session_state.responses.append(answer)
            utils.log_to_csv(query, answer)
            st.session_state.stream_handler.container.empty()
            if "query_start" in st.session_state:
                del st.session_state.prompt_key
                del st.session_state.query_start
    if "query_start" not in st.session_state: 
        with chat_container:
            display_chat_history()
def main():
    st.title("TalentMap 💬")
    if "INGESTION" in st.session_state:
        with st.spinner("Please wait ingestion in progress...."):
            if "INGESTION" not in st.session_state:
                os.sleep(1)               
    elif 'FOLLOW_UP' in st.session_state:
        inference_followup()        
    else:
        inference_start()

login_status = login()
if login_status == True:
    setUpSessionState()
    setUpSideBar()
    main()
