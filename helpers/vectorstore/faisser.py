from pathlib import Path
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from helpers.singleton.pickler import PickleOperator
from helpers.utils import hash_uuid


class FaissDB:
    def __init__(
        self,
        filename: str | Path = Path("vectorstore.pkl"),
        documents: list[Document] = [],
    ):
        self.filename = Path(filename)
        self.pickler = PickleOperator()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        # self.embeddings = HuggingFaceHubEmbeddings(huggingfacehub_api_token="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")
        self.vectorstore = self.load()
        if documents:
            self.add_documents(documents, write_to_disk=True)

    def save(self):
        self.pickler.dump(self.filename, self.vectorstore)

    def load(self, inplace: bool = False):
        vs: FAISS | None = self.pickler.load(self.filename)
        if inplace:
            self.vectorstore = vs
        return vs

    def _check_vectorstore(self):
        if self.vectorstore is None:
            raise Exception("Vectorstore not initialized! Pass in documents first.")

    def add_documents(self, docs: list[Document], write_to_disk: bool = False):
        docs = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata | {"content_hash": hash_uuid(doc.page_content).hex},
            )
            for doc in docs
        ]
        hashes = [d.metadata["content_hash"] for d in docs]
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            returned_docs = self.similarity_search(
                "", k=len(hashes), filter={"content_hash": hashes}
            )
            returned_hashes: set[int] = {x.metadata["content_hash"] for x in returned_docs}
            docs = list(filter(lambda x: x.metadata["content_hash"] not in returned_hashes, docs))
            if len(docs) > 0:
                self.vectorstore.add_documents(docs)
        if write_to_disk:
            self.save()

    def add_document(self, doc: Document, write_to_disk: bool = True):
        self.add_documents([doc], write_to_disk=write_to_disk)

    def delete_documents(self, ids: list[str], write_to_disk: bool = True):
        self._check_vectorstore()
        self.vectorstore.delete(ids)
        if write_to_disk:
            self.save()

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None):
        self._check_vectorstore()
        return self.vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict | None = None
    ):
        self._check_vectorstore()
        return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, filter: dict | None = None
    ):
        self._check_vectorstore()
        return self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=filter
        )
    
    def similarity_search_by_vector(
        self, embeddings: list[float], k: int = 4, filter: dict | None = None
    ):
        self._check_vectorstore()
        return self.vectorstore.similarity_search_by_vector(
            embeddings, k=k, filter=filter
        )
    
    def similarity_search_with_score_by_vector(
        self, embeddings: list[float], k: int = 4, filter: dict | None = None
    ):
        self._check_vectorstore()
        return self.vectorstore.similarity_search_with_score_by_vector(
            embeddings, k=k, filter=filter
        )

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, filter: dict | None = None
    ):
        self._check_vectorstore()
        return self.vectorstore.max_marginal_relevance_search(query, k=k, filter=filter)

    def merge(self, other: FAISS, write_to_disk: bool = True):
        self._check_vectorstore()
        vdb = (
            other
            if isinstance(other, FAISS)
            else other.vectorstore
            if isinstance(other, self.__class__)
            else None
        )
        if vdb is None:
            raise Exception("Invalid other vectorstore!")
        self.vectorstore.merge_from(other)
        if write_to_disk:
            self.save()

    def delete(self, write_to_disk: bool = True):
        self.vectorstore = None
        if write_to_disk:
            self.save()
    
    def delete_docs_by_page_content(self, docs: list[str|Document]) -> bool:
        page_contents = [x for x in docs if isinstance(x, str)]
        docs = [x for x in docs if isinstance(x, Document)]
        searched_docs = [self.similarity_search(x, k=1)[0] for x in page_contents]
        docs += searched_docs
        del searched_docs, page_contents
        doc_ids = [
            x for x in self.vectorstore.index_to_docstore_id.values()
            if self.vectorstore.docstore.search(x) in docs
        ]
        return self.vectorstore.delete(doc_ids)

    @property
    def total_documents(self):
        self._check_vectorstore()
        return len(self.vectorstore.index_to_docstore_id)