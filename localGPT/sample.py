import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import pdb
from langchain.llms import LlamaCpp
pdb.set_trace()
#MODEL_NAME = "TheBloke/Llama-2-13b-Chat-GPTQ"
#MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_NAME = "TheBloke/Llama-2-7b-Chat-GGUF"



llm = LlamaCpp( model_path=MODEL_NAME,
                n_gpu_layers=100,
                n_batch=512,
                n_ctx=4096,
                f16_kv=True,
                verbose=True,
                )

query = "show candidates suitable for role of front end developer"

result = llm("provide only skill as single word, required for below job description, in python list format for example role_skill=[skill1, skill2]. Show only role_skill list. " + query)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
)
 
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15
 
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)
 
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

query = "show candidates suitable for role of front end developer"

result = llm("provide only skill as single word, required for below job description, in python list format for example role_skill=[skill1, skill2]. Show only role_skill list. " + query)
print(result)
