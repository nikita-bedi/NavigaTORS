import os
import streamlit as st
from tqdm import tqdm
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
from transformers import AutoTokenizer, AutoModel
from langchain_openai import ChatOpenAI
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from langchain.docstore.document import Document
from llama_index.core import SimpleKeywordTableIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle, get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from typing import List, Any
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DATA_FOLDER = 'data' 
MODEL_NAME = "togethercomputer/m2-bert-80M-8k-retrieval"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
#HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


# Setup the service context with embedding and LLM model
def setup_service_context():
    embed_model = TogetherEmbedding(api_key=TOGETHER_API_KEY, model_name=MODEL_NAME)
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
    service_context = {
        'embed_model': embed_model,
        'llm': llm
    }
    return service_context

# Load documents and create VectorStoreIndex
def setup_document_index():
    documents = SimpleDirectoryReader(DATA_FOLDER).load_data()  # Loads data with doc_id
    service_context = setup_service_context()
    index = VectorStoreIndex.from_documents(
        documents,
        service_context={'embed_model': service_context['embed_model']},
        show_progress=True
    )
    return index, service_context
