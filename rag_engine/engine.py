from llama_index import (VectorStoreIndex, download_loader, ServiceContext)
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import HuggingFaceLLM
import torch
from pathlib import Path
import os
#Imports local prompts file
from .prompt import *
from .strategies import composable_strat

#
#Custom LLM
#


#We can do composable strategies in LlamaIndex with some modifications
#For example,
# chained_function = refineQuery(query) >> getSubQuestions >> Answer all subquestions >> summarize all answers
#TODO: Make some composable strategies here

#Strategy 1: Simple Query

#Strategy 2: Retrieve Context





#Making a custom QueryEngine
#This isn't super robust, but it should get the job done (RAG with LlamaIndex + any HuggingFace model)
#TODO: Make custom query/stream_query functions

class ElliotEngine:

    #Ray Serve will raise a warning if __init__ takes longer than 30 seconds. Don't worry, 99% of the time is to download the model from HuggingFace
    def __init__(self):
        #Check for OpenAI key, since we're
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("Please add the OPENAI_API_KEY environment variable to run this script. Run the following in your terminal `export OPENAI_API_KEY=...`")
        #Ingest the document, chunk it
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()

        #TODO: Change this to "Quick Python Guide" later
        docs = loader.load_data(file=Path('data/hello_world.pdf'))

        #Load LLM
        codegen_one = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.1, "do_sample": True},
            #This is the smallest CodeGen model I could find, makes deployment way easier
            tokenizer_name = "Salesforce/codegen-350M-mono",
            model_name = "Salesforce/codegen-350M-mono",
            device_map = "auto",
            tokenizer_kwargs={"max_length": 2048},
            #Need to do torch.float32 for CPU usage. For CUDA, we can do torch.float16
            model_kwargs={"torch_dtype": torch.float32}
        )

        #Create our ServiceContext, which bundles up our RAG assets
        service_context = ServiceContext.from_defaults(chunk_size = 512, llm=codegen_one)

        #Add to a VectorStore
        #Note: Ran into a bug while using a local HF embedding model, so we'll just default to OpenAI Embedding as a workaround
        vec_store: VectorStoreIndex = VectorStoreIndex.from_documents(docs, service_context = service_context)

        #This query engine will retrieve relevant context from vectorDB to use in a query
        #TODO: Erase this query engine and make my own
        self.qe: BaseQueryEngine = vec_store.as_query_engine(streaming=False)

    def query(self, prompt: str):
        answer = self.qe.query(prompt)
        return answer

    #Add query funcs

#On query,
# embed the query
# find relevant context
# return relevant context

if __name__ == "__main__":
    print("Trying to load the engine")
    new_engine = ElliotEngine()
    print(new_engine.query("Write a short Python program to add two floating point numbers"))
    print("Loaded the engine")