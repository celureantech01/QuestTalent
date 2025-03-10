import pdb
import torch
import re
import os
import logging
from numba import cuda
import subprocess
import streamlit as st
#from run_localGPT import load_model
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
#from langchain_openai import OpenAIEmbeddings
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
import pickle
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from haystack.nodes import PreProcessor
cache_file = "./linkedin_data"
import pdb
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

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
        with st.chat_message('assistant'):
            st.write(ent["text"])

        if count < len(st.session_state['requests']):
            with st.chat_message('user'):
                st.write(st.session_state['requests'][count])

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

with st.sidebar:
#    st.title("🤗💬 TalentMap")
    st.image('TalentMap.jpeg', caption='Making talent search easier', width=300 )
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
        logging.debug("Total results from SQL : " + str(len(dbList)))
        print("Try to get some documents from vector store also\n")
        vecList = []
        if 'RETRIEVER' in st.session_state:
            vecDoc = st.session_state.RETRIEVER.get_relevant_documents(query, callbacks=run_manager.get_child())
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
        return texts



import streamlit as st
from streamlit_login_auth_ui.widgets import __login__

__login__obj = __login__(auth_token = "pk_prod_N8PZ9A6BY446D0NSY3XAFVVJ26TA",
                    company_name = "Shims",
                    width = 200, height = 250,
                    logout_button_name = 'Logout', hide_menu_bool = False,
                    hide_footer_bool = False,
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()
if LOGGED_IN == True:
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
        DB = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=st.session_state["EMBEDDINGS"],
                client_settings=CHROMA_SETTINGS,
                )
        st.session_state.DB = DB

    if 'RETRIEVER' not in st.session_state:
        RETRIEVER = CustomVecRetriever(vectorstore=DB.as_retriever(search_kwargs={"k":2}))
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




    st.title("TalentMap 💬")

    st.session_state["SESSION_START"] = True
    lurl = "https://lottie.host/0423bb69-c510-4cbd-8e8b-feb2e76ed076/5E71JJzTC8.json"
    ltitle = load_lottieurl(lurl)
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

    if followup_reload != "Follow up":
        Skills = st.multiselect(label = "Choose a skill", options = uniqeskillList)

        workEx = st.text_input("Input work experience greater than", on_change=enable)

        Prompt = st.text_area("Input Job Description here", on_change=enable)
        if 'disabled' not in st.session_state:
            st.session_state.disabled = False
        submit=st.button("Submit Query", on_click=disable, disabled=st.session_state.disabled)
        st.session_state.START_SESSION = True


        final_answer = ""
        if submit:
            lottie_url_hr = "https://lottie.host/21f80e15-26ff-45ec-98d4-0585c131feae/eLe8LDFcfB.json"
            lottie_hr = load_lottieurl(lottie_url_hr)
            wait_lottie = "https://lottie.host/6c8a0848-269f-4523-89f0-067e2a8eb724/iiGOVTvVwP.json"
            wait_lottie_widget = load_lottieurl(wait_lottie)


#        with st_lottie_spinner(lottie_hr, key="progress", width =300, loop=True):
            with st.spinner("Please wait... (results will be streaming so you might need to scroll down the page)"):
                final_answer = ""
#            if 'QAStruct' not in st.session_state:
                if 'stream_handler' in st.session_state:
                    st.session_state.stream_handler.container = st.empty()
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
                        response = st.session_state['QAStruct']("Step by step go through each candidates details and find all candidates suitable for job mentioned next. " + Prompt + ". for each selected candidate explain in not more than 150 words why you have selected that candidate. Also show their linkedin url with name as a hyperlink. Also Show their git statistics.\n")
                    answer, docs = response["result"], response["source_documents"]
                if (('QA' in st.session_state) or (len(docs) == 0)):
                    if 'QA' not in st.session_state:
                        prompt, memory = model_memory()
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
                        response = st.session_state['QA']("Show all users. " + Prompt + ". for each selected candidate explain in not more than 200 words why you have selected that candidate. Also show their linkedin url with name as a hyperlink. Also Show their git statistics.")

                    answer, docs = response["result"], response["source_documents"]
                    final_answer = answer
                if 'resume' not in st.session_state:
#                st.write("** Currently subset of candidates are shown, you can check for new candidates in follow up questions e.g show me new candidates suitable for job apart from which you have already shown **")
                    resume = {}
                    for docent in docs:
                        name = re.findall(r"Resume of candidate \w+ *\w+", docent.page_content)
                        name = name[0].split('Resume of candidate ')[1]
                        if name not in resume.keys():
                            resume[name] = docent.page_content
# Get linkedin url from resume
                            obj = re.search(r"linkedin.com\/\w*\/\w*", resume[name])
                            if obj != None:
                                linkedin_url = obj.group()
                                linkedin_url = linkedin_url.lstrip("https://www.")
                                userDict = scrape_linkedin_profile(linkedin_url)
                                for keys in userDict.keys():
                                    resume[name] = resume[name] + "\n" + str(keys) + "\n" + str(userDict[keys])

                    st.session_state['resume'] = resume
                    st.session_state['extra_doc'] = docs
                    print("All resumes : " + str(st.session_state['resume']))
                while True:
                    response = st.session_state['QAStruct']("show me new candidates suitable for job apart from which you have already shown. Make sure you are not showing candidates which you have already selected, also continue the count from where it was left. If there are no new candidate say Above all are selected candidates.")
                    answer, docs = response["result"], response["source_documents"]
                    obj = re.search(r"Above all are selected candidates", answer)
                    if obj != None:
                        break
                    final_answer = final_answer + "\n"+ answer
                
                if 'stream_handler' in st.session_state:
                    st.session_state.stream_handler.container.empty()
                    st.session_state.stream_handler.text = ""
                if 'resume' not in st.session_state:
                    st.subheader("👉 " + "TalentMap Recommendations: ")
                else:
                    st.subheader("👉 " + "TalentMap:")

                st.write(final_answer)
            enable()
            st.success("Hope I was able to save your time")

        if 'ANS' not in st.session_state:
            if answer != '':
                st.session_state['ANS'] = final_answer
                st.session_state['responses'].append(final_answer)
                st.session_state['requests'].append(Prompt)
                st.session_state.FOLLOW_UP = "Follow up"

    else:
        
        chat_container = st.container()
        prompt_container = st.container()
        with chat_container:
            display_chat_history()

        with prompt_container:
            if 'stream_handler' in st.session_state:
                st.session_state.stream_handler.container = chat_container
            query = st.text_input('Prompt: ', placeholder='Enter your prompt here..')
            answer = ''
            if query:
                with st.spinner('Generating Response...'):
                    response = st.session_state['QAStruct'](query)
                    answer, docs = response["result"], response["source_documents"]
                st.session_state.requests.append(query)
                st.session_state.responses.append(answer)
