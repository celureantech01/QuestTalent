import torch
import re
import os
import logging
from numba import cuda
import subprocess
import streamlit as st
from run_localGPT import load_model
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from prompt_template_utils import get_prompt_template
from langchain_openai import OpenAIEmbeddings
from streamlit_lottie import st_lottie_spinner
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import pdb
from langchain.schema.retriever import BaseRetriever
#from langchain.schema.retriever import VectorStoreRetriever
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.document import Document
from typing import List
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.vectorstores import VectorStore
import logging.handlers
import threading

def scrape_linkedin_profile(linkedin_profile_url: str):
#    """
#    Getting Linkedin data using nubela's proxy curl api
#    """
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"

    response = requests.get(
                api_endpoint,
                params = {"url": linkedin_profile_url},
                headers = {"Authorization": f"Bearer {os.getenv('PROXY_CURL_API')}"}
                )

    data = response.json()
    pdb.set_trace()
#    data = {
#            key: value
##            for key, value in data.items()
##            if key in include_keys and value not in ([], "", None,'')
##            }
    return data
#    return ""

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
    root.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
    root.addHandler(handler)


setLoggingHandler()
answer=''
logging.basicConfig(level=logging.DEBUG)
if 'FOLLOW_UP' in st.session_state:
    followup_reload = st.session_state.FOLLOW_UP
else:
    followup_reload = ''
def decideFollowOrReload(option_select):
    print("Selected option is :" + str(option_select))

def reload():
    del st.session_state['ANS'] 
    del st.session_state.disabled
    del st.session_state.SQL_LLM
    del st.session_state.EMBEDDINGS
    del st.session_state.DB
    del st.session_state.RETRIEVER
    del st.session_state.CUSTOM_RETRIEVER
    del st.session_state["LLM"]
    del st.session_state["SQL_LLM"]
    del st.session_state["SESSION_START"]
    del st.session_state.QAStruct
    del st.session_state['resume']
    st.rerun()
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


#def model_memory():
#    prompt, memory = get_prompt_template(promptTemplate_type="llama", history=True)
#    return prompt, memory

def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. Answers should be specific to Question asked. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer. Consider resume and profile as same.\

    {context} 

    {history}
    Question: {question}
    Helpful Answer:"""
    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferWindowMemory(input_key="question", memory_key="history", k=5)

    return prompt,memory

with st.sidebar:
#    st.title("🤗💬 TalentMap")
    st.image('/home/ubuntu/localGPT/TalentMap.jpeg', caption='Making talent search easier', width=300 )
    st.markdown(
        """
    ## About
    This is Talent search app. 
 
    """
    )
    add_vertical_space(5)

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
            documents = self.vectorstore.get_relevant_documents(query, callbacks=run_manager.get_child())
            return documents


class CustomRetriever(BaseRetriever):
    sQuery: dict = {}
#    @cuda.jit
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        if 'resume' in st.session_state:
            logging.debug("Inside Follow up for first time SQL. Num of docs: " + str(len(st.session_state.resume)))
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

        db = SQLDatabase.from_uri("sqlite:///database1.db", max_string_length=6000)
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
            query = query + order_by_text + " ) DESC LIMIT 20;"
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
            pdb.set_trace()
            try:
                data = scrape_linkedin_profile(linkdinURL)
            except:
                pass

        logging.debug("Total results from SQL : " + str(len(dbList)))
        text_splitter = RecursiveCharacterTextSplitter(
#                       chunk_size=3000,
                       chunk_size=100000,
                        chunk_overlap=100,
                        length_function=len,
                        is_separator_regex=False,
                        )
        texts = text_splitter.create_documents(dbList)
        return texts


if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"
if "EMBEDDINGS" not in st.session_state:
    EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
    st.session_state.EMBEDDINGS = EMBEDDINGS

if "DB" not in st.session_state:
    DB = Chroma(
       persist_directory=PERSIST_DIRECTORY,
        embedding_function=st.session_state["EMBEDDINGS"],
        client_settings=CHROMA_SETTINGS,
    )
    st.session_state.DB = DB

if 'RETRIEVER' not in st.session_state:
    RETRIEVER = CustomVecRetriever(vectorstore=DB.as_retriever(search_kwargs={"k":20}))
#    RETRIEVER = DB.as_retriever(search_kwargs={"k":3})
    st.session_state.RETRIEVER = RETRIEVER

if 'CUSTOM_RETRIEVER' not in st.session_state:
    CUSTOM_RETRIEVER = CustomRetriever()
    st.session_state.CUSTOM_RETRIEVER = CUSTOM_RETRIEVER
if "LLM" not in st.session_state:
#    LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
#    query = "show candidates suitable for role of front end developer"

#    result = LLM("provide only skill as single word, required for below job description, in python list format for example role_skill=[skill1, skill2]. Show only role_skill list. " + query)
    SQL_LLM = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    LLM = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
    st.session_state["LLM"] = LLM
    st.session_state["SQL_LLM"] = LLM




st.title("TalentMap 💬")

if True:
    st.session_state["SESSION_START"] = True
    lurl = "https://lottie.host/0423bb69-c510-4cbd-8e8b-feb2e76ed076/5E71JJzTC8.json"
    ltitle = load_lottieurl(lurl)
    df = pd.read_csv("../backup/9_02.csv")
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

    if followup_reload != "Follow up":
        Skills = st.multiselect(label = "Choose a skill", options = uniqeskillList)

        workEx = st.text_input("Input work experience greater than", on_change=enable)

        Prompt = st.text_area("Input Job Description here", on_change=enable)
        if 'disabled' not in st.session_state:
            st.session_state.disabled = False
        submit=st.button("Submit Query", on_click=disable, disabled=st.session_state.disabled)
        st.session_state.START_SESSION = True

    else:
        Prompt = st.text_area("Follow up query on selected results", on_change=enable)
        if 'disabled' not in st.session_state:
            st.session_state.disabled = False
        submit=st.button("Submit Query", on_click=disable, disabled=st.session_state.disabled)
        with st.expander("Previous Results"):
            if 'ANS' in st.session_state:
                st.write(st.session_state.ANS)

    if submit:
        lottie_url_hr = "https://lottie.host/21f80e15-26ff-45ec-98d4-0585c131feae/eLe8LDFcfB.json"
        lottie_hr = load_lottieurl(lottie_url_hr)
        wait_lottie = "https://lottie.host/6c8a0848-269f-4523-89f0-067e2a8eb724/iiGOVTvVwP.json"
        wait_lottie_widget = load_lottieurl(wait_lottie)


        with st_lottie_spinner(lottie_hr, key="progress", width =300, loop=True):
#            if 'QAStruct' not in st.session_state:
            if followup_reload != "Follow up":
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
            else:
                pass

            if 'QA' not in st.session_state:
                if 'FOLLOW_UP' in st.session_state:
                    response = st.session_state['QAStruct'](Prompt)
                else:
                    response = st.session_state['QAStruct']("Show all users. " + Prompt + ". Explain in 2-3 lines why you have selected these users. Also show their linkedin url with name as a hyperlink. Also Show their git statistics. Show end summary not more than 200 words, also try to highlight important datapoints relating to job. Also provide there full name as Full_Name: terminated by new line character, for example if full name is ABC DEF then provide name as Full_Name: ABC DEF\n")
                answer, docs = response["result"], response["source_documents"]
            if (('QA' in st.session_state) or (len(docs) == 0)):
                if 'QA' not in st.session_state:
                    QA = RetrievalQA.from_chain_type(
                        st.session_state['LLM'],
                        chain_type="stuff",
#                        chain_type="map_reduce",
                        retriever=st.session_state['RETRIEVER'],
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt, "memory": memory},
                        )
                    st.session_state['QA'] = QA
                if 'FOLLOW_UP' in st.session_state:
                    response = st.session_state['QA'](Prompt)
                else:
                    response = st.session_state['QA']("Show all users. " + Prompt + ". Explain in 2-3 lines why you have selected these users. Also show their linkedin url with name as a hyperlink. Also Show their git statistics. Show end summary not more than 200 words, also try to highlight important datapoints relating to job. Also provide there full name as Full_Name: terminated by new line character, for example if full name is ABC DEF then provide name as Full_Name: ABC DEF\n")
                answer, docs = response["result"], response["source_documents"]
            if 'resume' not in st.session_state:
                st.subheader("👉 " + "TalentMap Recommendations:")
            else:
                st.subheader("👉 " + "TalentMap:")

            st.write(answer)
            if 'resume' not in st.session_state:
                resume = {}
                nameList = re.findall(r"Full_Name: \w+ *\w*", answer)
                for ent in nameList:
                    name = ent.split("Full_Name: ")[1]
                    resume[name] = [i.page_content for i in docs if name in i.page_content][0]
                st.session_state['resume'] = resume
                print("All resumes : " + str(st.session_state['resume']))
        enable()
        st.success("Hope I was able to save your time")
followup_reload = st.radio("Click for follow up queries on current results, or refresh for new query",
                         ["current", "Follow up"],
                      captions = ["current view", "chat based on selected results"], on_change=enable)
#st.button("GO", on_click=enable)
Stseperator = "\n\n---------------------START---------------------------------\n\n"
Endseperator = "\n\n--------------------END-----------------------------------\n\n"

if 'ANS' not in st.session_state:
    if answer != '':
        st.session_state['ANS'] = answer
else:
    if answer != '':
        st.session_state['ANS']  = Stseperator + answer + Endseperator + st.session_state['ANS']

if followup_reload == "Follow up":
#    pdb.set_trace()
    if 'FOLLOW_UP' not in st.session_state:
        st.session_state.FOLLOW_UP = "Follow up"
        st.rerun()
    elif followup_reload == "Refresh":
        reload()
else:
     pass
