import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

context_format = """
### Source: {source} 
### Page:{page}
{content}
"""


class ContextManager:
    def __init__(self, config=None):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True,
        )

        model_kwargs = {"device": "cuda"}
        self.embeddings = HuggingFaceEmbeddings(model_kwargs=model_kwargs)
        self.vectordb = Chroma(
            persist_directory="doc_index", embedding_function=self.embeddings
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 25})

    def get_sources(self):
        sources = map(
            lambda x: os.path.basename(x["source"]),
            self.vectordb._collection.get()["metadatas"],
        )
        sources_list = list(set(sources))
        sources_list.sort()
        return "\n".join(sources_list)

    def format_context(self, context):
        output = ""
        seen = set()
        for doc in context:
            source = os.path.basename(doc.metadata["source"])
            page = doc.metadata["page"]
            content = doc.page_content
            if hash(content) not in seen:
                output += context_format.format(
                    source=source, page=page, content=content
                )
                seen.add(hash(content))
        return output

    def get_context(self, question):
        context = self.retriever.get_relevant_documents(question)
        print(context)
        return self.format_context(context)

    def upload_file(self, files):
        file_paths = [file.name for file in files]
        for file_path in file_paths:
            loader = PyPDFLoader(file_path, extract_images=False)
            pages = loader.load_and_split()
            chunks = self.text_splitter.split_documents(pages)
            db = Chroma.from_documents(
                chunks, embedding=self.embeddings, persist_directory="doc_index"
            )
            db.persist()
        return file_paths
