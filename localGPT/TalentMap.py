import torch
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
from prompt_template_utils import get_prompt_template
from langchain_openai import OpenAIEmbeddings
from streamlit_lottie import st_lottie_spinner
from streamlit_lottie import st_lottie
import requests
import pandas as pd

def disable():
    st.session_state.disabled = True

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.\
    Always try to summarize reponse in less than 200 words.

    {context}

    Question: {question}
    Helpful Answer:"""
#    pdb.set_trace()
#    prompt, memory = get_prompt_template("mistral")
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, ""


# Sidebar contents
with st.sidebar:
#    st.title("🤗💬 TalentMap")
    st.image('/home/ubuntu/localGPT/TalentMap.jpeg', caption='Making talent search easier', width=300 )
    st.markdown(
        """
    ## About
    This app is an LLM-powered TalentMapping framework:
 
    """
    )
    add_vertical_space(5)

if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"


# if "result" not in st.session_state:
#     # Run the document ingestion process.
#     run_langest_commands = ["python", "ingest.py"]
#     run_langest_commands.append("--device_type")
#     run_langest_commands.append(DEVICE_TYPE)

#     result = subprocess.run(run_langest_commands, capture_output=True)
#     st.session_state.result = result

# Define the retreiver
# load the vectorstore
import pdb
#pdb.set_trace()
#if "EMBEDDINGS" not in st.session_state:
EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
#EMBEDDINGS=OpenAIEmbeddings()

#if "DB" not in st.session_state:
DB = Chroma(
       persist_directory=PERSIST_DIRECTORY,
#        embedding_function=st.session_state["EMBEDDINGS"],
        embedding_function=EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
    )

from langchain.schema.retriever import BaseRetriever
from langchain.schema.document import Document
from typing import List
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.vectorstores import VectorStore

def getDbResultOnly(query):
    import pdb
#    pdb.set_trace()
    db = SQLDatabase.from_uri("sqlite:///database1.db", max_string_length=6000)
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
#    llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True, top_k=200)

    custom_query = "from table userDB show only resume column of candidates. Kindly do case insensitive search. Return only source code " + query
    output = agent_executor.invoke(custom_query)
#    dbquery=output['output'].split("sql\n")[1].split(";\n")[0]
#    dbOut=db.run(dbquery)
    dbOut = output['output']
    return dbOut

class CustomRetriever(BaseRetriever):
    sQuery: dict
#    vectorstores:VectorStore
#    retriever:db.as_retriever()
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
#        pdb.set_trace()
        db = SQLDatabase.from_uri("sqlite:///database1.db", max_string_length=6000)
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
#        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True, top_k=50)

        skills = self.sQuery['skills']
        workexp = self.sQuery['workEx']
        skill_text = ''
        for sk in skills:
            skill_text = skill_text + sk +  ' '

        if skill_text == '' and workexp == '':
            text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=100000,
                        chunk_overlap=500,
                        length_function=len,
                        is_separator_regex=False,
                    )

            texts = text_splitter.create_documents([])
            return texts

        output = agent_executor.invoke({"input": "Construct sql query using schema of userDB table for user input and show only top 50 resume of users. Make sure to show only resume column from userDB table. Don't execute query. Put 'sql\n' at start of query and ';\n' at end of sql query. In userDB table show only resume column of users having skills in " + skill_text + ". Also work experience should be greater than " + workexp})        
#        query = query.split("<U>")[1].split("</U>")[0]
        
#        output = agent_executor.invoke("Only for role in user input find skills from role_skill_table. Just show only one top entry and make case insensitive serach. In output don't display anything from userDB table or mention userDB table. " + self.sQuery)
#        custom_query = '''
#        construct sql query using schema of userDB table for user input and show only top 50 resume of users.\
#Make sure to show only resume column from userDB table.\
#In query ignore Job Title field from userDB table. \
#Always try to construct case insensitive query. Job title is not mandatory condition.\
#Do note in schema some columns have spaces in there name, so you need to take care of spaces while constructing query. Don't execute query.\
#Put 'sql\n' at start of query and ';\n' at end of sql query " 
#        '''
#        custom_query = custom_query + "\n Skills: \n" + output['output'] +  "\n user input: " + self.sQuery
#        custom_query = custom_query + "\n user input: " + self.sQuery
#        custom_query = "construct sql query using schema of userDB and role_skill_table table for user input and show only top 50 resume of users. For role in user input, try to find skills in role_skill_table. Using skills from role_skill_table in combination with skills provided in user input, do a 'in' search for skills in userDB table. While searching from role_skill_table do case insesitive search. Remember skills we got from role_skill_table should be searched in userDB table but it is not required that those skills match in userDB. Don't consider role or job title field while doing a query from userDB table. Make sure to show only resume column from userDB table. Always try to construct case insensitive query. Do note in schema some columns have spaces in there name, so you need to take care of spaces while constructing query. Don't execute query.  Put 'sql\n' at start of query and ';\n' at end of sql query " + self.sQuery
#        custom_query = "from table userDB show only resume column of candidates. " + self.sQuery
#        output = agent_executor.invoke(custom_query)
        dbquery=output['output'].split("```sql\n")[1].split("```\n")[0]
        dbOut=db.run(dbquery)
#        dbOut = output['output']
        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=100000,
                        chunk_overlap=500,
                        length_function=len,
                        is_separator_regex=False,
                        )
        texts = text_splitter.create_documents([dbOut])
        return texts


import pdb
#pdb.set_trace()
#if "RETRIEVER" not in st.session_state:
#RETRIEVER = DB.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k":10})
#RETRIEVER = DB.as_retriever(search_type="mmr", search_kwargs={"k":30, "fetch_k":100})
RETRIEVER = DB.as_retriever(search_kwargs={"k":5})
customRETRIEVER = CustomRetriever(sQuery="")
#st.session_state["RETRIEVER"] = RETRIEVER

if "LLM" not in st.session_state:
#    LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
#    LLM = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    LLM = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
#    st.session_state["LLM"] = LLM


if "QA" not in st.session_state:
    prompt, memory = model_memory()

    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="map_reduce",
        retriever=RETRIEVER,
        return_source_documents=True,
        chain_type_kwargs={"question_prompt": prompt},
    )
    st.session_state["QA"] = QA

    QAStruct = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="map_reduce",
        retriever=customRETRIEVER,
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={"question_prompt": prompt},
    )
    st.session_state["QA"] = QA
    st.session_state["QAStruct"] = QAStruct


#st.set_page_config(layout="wide")
# Custom HTML/CSS for the banner
#st.image('/home/ubuntu/localGPT/TalentMap.jpeg', caption='Making talent search easier', width=300)
#st.title("TalentMap App 💬")
st.title("TalentMap 💬")
# Create a text input box for the user

lurl = "https://lottie.host/0423bb69-c510-4cbd-8e8b-feb2e76ed076/5E71JJzTC8.json"
ltitle = load_lottieurl(lurl)
from streamlit_lottie import st_lottie
#st_lottie(ltitle)
#st.title("TalentMap")
#st.image('/home/ubuntu/localGPT/TalentMap.jpeg', caption='Making talent search easier', width=300)

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

Skills = st.multiselect(label = "Choose a skill", options = uniqeskillList)


workEx = st.text_input("Input work experience greater than")

Prompt = st.text_area("Input Job Description here")
if "disabled" not in st.session_state:
    st.session_state.disabled = False

# while True:
submit=st.button("Submit Query", on_click=disable, disabled=st.session_state.disabled)


# If the user hits enter
if submit:
    from streamlit_lottie import st_lottie_spinner
    lottie_url_hr = "https://lottie.host/21f80e15-26ff-45ec-98d4-0585c131feae/eLe8LDFcfB.json"
    lottie_hr = load_lottieurl(lottie_url_hr)
#    pdb.set_trace()
    wait_lottie = "https://lottie.host/6c8a0848-269f-4523-89f0-067e2a8eb724/iiGOVTvVwP.json"
    wait_lottie_widget = load_lottieurl(wait_lottie)
#    st_lottie(wait_lottie_widget)
    with st_lottie_spinner(lottie_hr, key="progress", width =300, loop=True):
#        st.write("👉 :blue[Searching...]")
#    with st.spinner("👉 " + "AI Response: Please wait..."):
    # Then pass the prompt to the LLM
        prompt, memory = model_memory()


#        pdb.set_trace()
        skills_work = {'skills': Skills,
                        'workEx': workEx}

        QAStruct = RetrievalQA.from_chain_type(
        llm=LLM,
#        chain_type="map_reduce",
        chain_type="stuff",
        retriever=CustomRetriever(sQuery=skills_work),
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={"prompt": prompt},
        )
        db = SQLDatabase.from_uri("sqlite:///database1.db", max_string_length=6000)
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True, top_k=50)
#        output = agent_executor.invoke("Only for role in user input find skills from role_skill_table. Just show only one top entry and make case insensitive serach. In output don't display anything from userDB table or mention userDB table. " + Prompt)
#        pdb.set_trace()
#        Prompt = Prompt + "\nAdditional skills : " + output['output']
#        response = QAStruct("Show all users. " + Prompt + ". Tell me reason why you have selected them and show concise summary of their resume. Also show there linkedin url and git contributions. If some users have git stars more than 5 show star in front of there name")
        response = QAStruct("Show all users. " + Prompt + ". Tell me reason why you have selected them and show concise summary of their resume. Also show their linkedin url as a hyperlink with name. If some users have git repositories greater than 5, show yellow star after their name. Also put text in  end summary in bold ")
        answer, docs = response["result"], response["source_documents"]
#        pdb.set_trace()
        if (len(docs) == 0):
            QA = RetrievalQA.from_chain_type(
                llm=LLM,
                chain_type="map_reduce",
                retriever=RETRIEVER,
                return_source_documents=True,
                chain_type_kwargs={"question_prompt": prompt},
                )

#            response = QA("Show all users. " + Prompt + ". Tell me reason why you have selected them and show concise summary of their resume. Also show there linkedin url as a hyperlink, and git contributions. If some users have git stars more than 5 show bright star after there name. Also put recommendations and end summary in bold ")
            response = QA("Show all users. " + Prompt + ". Tell me reason why you have selected them and show concise summary of their resume. Also show there linkedin url as a hyperlink with name. If some users have git repositories greater than 5, show yellow star after their name. Also put text in  end summary in bold ")
            answer, docs = response["result"], response["source_documents"]

#        nquery = "<S>"+structPrompt+"</S><U>" + unStructPrompt + "</U>"
#        if (structPrompt != '' and unStructPrompt != ''):
#            response = QAStruct(unStructPrompt)
#            answer, docs = response["result"], response["source_documents"]
#        if (structPrompt != '' and unStructPrompt == ''):
#            answer = getDbResultOnly(structPrompt)
#        if (structPrompt == '' and unStructPrompt != ''):
#            response = QA(unStructPrompt)
#            answer, docs = response["result"], response["source_documents"]

#        pdb.set_trace()
#        answer, docs = response["result"], response["source_documents"]
     # ...and write it out to the screen
        st.subheader("👉 " + "TalentMap Recommendations:")
        st.write(answer)
        st.success("Hope I was able to save your time")
    # With a streamlit expander
#    with st.expander("Document Similarity Search"):
#        # Find the relevant pages
#        search = DB.similarity_search_with_score(prompt)
#        # Write out the first
#        for i, doc in enumerate(search):
#            # print(doc)
#            st.write(f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
#            st.write(doc[0].page_content)
####            st.write("--------------------------------")
