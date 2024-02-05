import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

context_format = """
### Context Rank: {rank_num}
### Source: {source} 
### Page:{page}
{content}
"""


class ContextManager:
    def __init__(self, config):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["text_splitter"]["chunk_size"],
            chunk_overlap=config["text_splitter"]["chunk_overlap"],
            length_function=len,
            add_start_index=True,
        )

        model_kwargs = {"device": config["embedding"]["device"]}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model"], model_kwargs=model_kwargs
        )
        self.k = config["embedding"]["max_docs"]
        self.vectordb = Chroma(
            persist_directory=f"doc_index/{config['embedding']['vectordb']}",
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": self.k})

    def change_db(self, db_name):
        self.vectordb = Chroma(
            persist_directory=f"doc_index/{db_name}", embedding_function=self.embeddings
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": self.k})

    def get_sources(self):
        sources = map(
            lambda x: os.path.basename(x["source"]),
            self.vectordb._collection.get()["metadatas"],
        )
        sources_list = list(set(sources))
        sources_list.sort()
        return "\n".join(sources_list)

    def format_context(self, context):
        matches = []
        output = ""
        seen = set()
        rank_num = 0
        for doc in context:
            source = os.path.basename(doc.metadata["source"])
            page = doc.metadata["page"]
            content = doc.page_content
            if hash(content) not in seen:
                rank_num += 1
                matches.append([source, page, content, rank_num])
                seen.add(hash(content))
        matches.sort(key=lambda x: x[0])
        for match in matches:
            source, page, content, rank_num = match
            output += context_format.format(
                source=source, page=page, content=content, rank_num=rank_num
            )
        return output

    def get_context(self, question):
        context = self.retriever.get_relevant_documents(question)
        return self.format_context(context)

    def upload_file(self, files, db_name=None):
        if db_name is None:
            db_name = "default"
        file_paths = [file.name for file in files]
        for file_path in file_paths:
            loader = PyPDFLoader(file_path, extract_images=False)
            pages = loader.load_and_split()
            chunks = self.text_splitter.split_documents(pages)
            db = Chroma.from_documents(
                chunks,
                embedding=self.embeddings,
                persist_directory=f"doc_index/{db_name}",
            )
            db.persist()
        return file_paths
