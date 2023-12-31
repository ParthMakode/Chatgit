import git
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.ollama import Ollama
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import GPT4AllEmbeddings
from queue import Queue

gpt4all_embd = GPT4AllEmbeddings()

allowed_extensions = ['.py', '.ipynb', '.md','.js','.jsx','.ts','.tsx']

class Embedder:
    def __init__(self, git_link) -> None:
        self.git_link = git_link
        last_name = self.git_link.split('/')[-1]
        self.clone_path = last_name.split('.')[0]
        self.model= Ollama(model="mistral", temperature=0,verbose=True,num_gpu=1,num_ctx=8000,)
        self.embed = gpt4all_embd
        self.Chatqueue =  Queue(maxsize=4)

    def add_to_queue(self, value):
        if self.Chatqueue.full():
            self.Chatqueue.get()
        self.Chatqueue.put(value)

    def clone_repo(self):
        if not os.path.exists(self.clone_path):
            # Clone the repository
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try: 
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e: 
                        pass
    
    def chunk_files(self):
        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=50)
        self.texts = text_splitter.split_documents(self.docs)
        self.num_texts = len(self.texts)

    def embed_chunks(self):
        db=FAISS.from_documents(self.texts,embedding=gpt4all_embd)
        
        db.save_local(folder_path="repo_index",index_name="repo_index1")
        
        return db
    
    def delete_directory(self, path):
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path)
        
    def load_db(self):
        self.extract_all_files()
        self.chunk_files()
        self.db = self.embed_chunks()

        self.retriever = self.db.as_retriever()
        self.retriever.search_kwargs['distance_metric'] = 'cos'
        self.retriever.search_kwargs['fetch_k'] = 100
        self.retriever.search_kwargs['maximal_marginal_relevance'] = True
        self.retriever.search_kwargs['k'] = 3


    def retrieve_results(self, query):
        chat_history = list(self.Chatqueue.queue)
        qa = ConversationalRetrievalChain.from_llm(self.model, chain_type="stuff", retriever=self.retriever, 
        condense_question_llm =self.model)
        
        result = qa({"question": query, "chat_history": chat_history})
        self.add_to_queue((query, result["answer"]))
        return result['answer']
