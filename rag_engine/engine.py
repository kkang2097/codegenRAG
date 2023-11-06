from llama_index import (VectorStoreIndex, download_loader, ServiceContext)
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import HuggingFaceLLM, CompletionResponse
import torch
from pathlib import Path
import os
#Imports local prompts file. When running this file as a script, comment this out.
#In deployment, we can import these files
# from .prompt import *
# from .strategies import composable_strat

#
#Custom LLM
#


#ServiceContext contains:
#llm - LLM object that we can use to query directly
#


#We can do composable strategies in LlamaIndex with some modifications
#For example,
# chained_function = refineQuery(query) >> getSubQuestions >> Answer all subquestions >> summarize all answers
#TODO: Make some composable strategies here

#Strategy 1: Simple Query

#Strategy 2: Retrieve Context
'''
@composable_strat
def retriever(query: str, context: ServiceContext):
    

    #Returns tuple of [query_result, serviceContext]
    return
'''



#Making a custom QueryEngine
#This isn't super robust, but it should get the job done (RAG with LlamaIndex + any HuggingFace model)
#TODO: Make custom query/stream_query functions

class ElliotEngine:

    #Ray Serve will raise a warning if __init__ takes longer than 30 seconds. Don't worry, 99% of the time is to download the model from HuggingFace
    def __init__(self):
        #Check for OpenAI key, since we need to embed documents for RAG
        #Edit: Since RAG hasn't been fleshed out yet, don't need OPENAI_API_KEY
        # if "OPENAI_API_KEY" not in os.environ:
        #     raise RuntimeError("Please add the OPENAI_API_KEY environment variable to run this script. Run the following in your terminal `export OPENAI_API_KEY=...`")

        #Ingest the document, chunk it
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()

        #TODO: Change this to "Quick Python Guide" later
        curr_dir = os.path.split(os.getcwd())[1]
        data_filepath = ""

        #Hacky conditional to handle initializing engine.py OR app.py
        if(curr_dir == "rag_engine"):
            data_filepath =os.getcwd() + '/../data/hello_world.pdf'
        else:
            data_filepath = os.getcwd() + '/data/hello_world.pdf'
        docs = loader.load_data(file=Path(data_filepath))

        #Load LLM
        self.codegen_one = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.1, "do_sample": True},
            #This is the smallest CodeGen model I could find, makes deployment way easier
            tokenizer_name = "Salesforce/codegen-350M-multi",
            model_name = "Salesforce/codegen-350M-multi",
            device_map = "auto",
            tokenizer_kwargs={"max_length": 2048},
            #Need to do torch.float32 for CPU usage. For CUDA, we can do torch.float16
            model_kwargs={"torch_dtype": torch.float32}
        )

        #Create our ServiceContext, which bundles up our RAG assets
        service_context = ServiceContext.from_defaults(chunk_size = 512, llm=self.codegen_one)

        #Add to a VectorStore
        #Note: Ran into a bug while using a local HF embedding model, so we'll just default to OpenAI Embedding as a workaround
        # vec_store: VectorStoreIndex = VectorStoreIndex.from_documents(docs, service_context = service_context)

        #TODO: Make custom query engine

    def query(self, prompt: str) -> str:
        answer: CompletionResponse = self.codegen_one.complete(prompt)
        #TODO: We'll need to condense the number of choices down later, since CodeGen returns multiple results
        #For example, we need to load in the composable strategies to filter down the output
        return answer

    #Add query funcs

#On query,
# embed the query
# find relevant context
# return relevant context

if __name__ == "__main__":
    new_engine = ElliotEngine()
    print(new_engine.query("# Write a Python function to append a string in the middle of another string"))