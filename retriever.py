import os
import re
import requests
import numpy as np
from google import genai
from dotenv import load_dotenv
import PyPDF2
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize

load_dotenv()

client = genai.Client(api_key=os.environ.get('GEMINI_KEY'))
from langchain_core.tools import tool

response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = client

    @classmethod
    def from_docs(cls, docs, client):
        embeddings = client.models.embed_content(
        model="models/text-embedding-004",
        contents=[doc["page_content"] for doc in docs])
        vectors = [emb.values for emb in embeddings.embeddings]
        return cls(docs, vectors, client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.models.embed_content(
            model="models/text-embedding-004",
            contents=query
        )
        
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.embeddings[0].values) @ self._arr.T
        
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        
        results = [
            {**self._docs[idx], "similarity": float(scores[idx])} for idx in top_k_idx_sorted
        ]
        return results
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def semantic_chunk_text(self, text: str, min_chunk_size: int = 200, max_chunk_size: int = 2000) -> List[str]:
        """Chunk text semantically by grouping related sentences together."""
        # First split into sentences
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Process sentences in batches for semantic similarity
        batch_size = 40  # Keep well under the 80 limit to account for current chunk
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # If we have a current chunk, add it to the batch for comparison
            if current_chunk:
                batch = [" ".join(current_chunk)] + batch
            
            # Get embeddings for the batch
            try:
                embeddings = self._client.models.embed_content(
                    model="models/text-embedding-004",
                    contents=batch
                )
                batch_embeddings = [np.array(emb.values) for emb in embeddings.embeddings]
                
                # If we have a current chunk, compare with first sentence in batch
                if current_chunk:
                    current_embed = batch_embeddings[0]
                    next_embed = batch_embeddings[1]
                    similarity = np.dot(current_embed, next_embed) / (
                        np.linalg.norm(current_embed) * np.linalg.norm(next_embed)
                    )
                    
                    # If similarity is low, save current chunk and start new one
                    if similarity < 0.7:
                        chunk_text = " ".join(current_chunk)
                        if len(chunk_text) >= min_chunk_size:
                            chunks.append(chunk_text)
                        current_chunk = []
                        current_size = 0
                
                # Process remaining sentences in batch
                for j, sentence in enumerate(batch[1:] if current_chunk else batch):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_size = len(sentence)
                    
                    # Check if adding this sentence would exceed max_chunk_size
                    if current_size + sentence_size > max_chunk_size and current_chunk:
                        chunk_text = " ".join(current_chunk)
                        if len(chunk_text) >= min_chunk_size:
                            chunks.append(chunk_text)
                        current_chunk = []
                        current_size = 0
                    
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_size += sentence_size
                    
                    # If we've reached min_chunk_size, check similarity with next sentence
                    if current_size >= min_chunk_size and j < len(batch) - 2:
                        current_embed = batch_embeddings[j + 1]
                        next_embed = batch_embeddings[j + 2]
                        similarity = np.dot(current_embed, next_embed) / (
                            np.linalg.norm(current_embed) * np.linalg.norm(next_embed)
                        )
                        
                        if similarity < 0.7:
                            chunk_text = " ".join(current_chunk)
                            chunks.append(chunk_text)
                            current_chunk = []
                            current_size = 0
                            
            except Exception as e:
                # If embedding fails, just use the current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) >= min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_size = 0
        
        # Add any remaining text as the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks

    def process_pdf_folder(self, folder_path: str) -> List[Dict]:
        """Process all PDF files in a folder and return a list of documents."""
        documents = []
        
        # Get all PDF files in the folder
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        print(f"\nProcessing {len(pdf_files)} PDF files from {folder_path}")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            try:
                print(f"\nProcessing {pdf_file}")
                # Extract text from PDF
                text = self.extract_text_from_pdf(pdf_path)
                print(f"Extracted {len(text)} characters")
                
                # Split text into chunks using semantic chunking
                chunks = self.semantic_chunk_text(text)
                print(f"Created {len(chunks)} chunks")
                
                # Create documents with metadata
                for i, chunk in enumerate(chunks):
                    doc = {
                        "page_content": chunk,
                        "metadata": {
                            "source": pdf_file,
                            "chunk": i + 1,
                            "total_chunks": len(chunks),
                            "chunk_size": len(chunk)
                        }
                    }
                    documents.append(doc)
                    
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                
        print(f"\nTotal documents created: {len(documents)}")
        return documents

    def insert_pdfs_to_vectorstore(self, folder_path: str) -> None:
        """Process PDFs and insert them into the VectorStore."""
        # Process PDFs
        documents = self.process_pdf_folder(folder_path)
        
        # Get embeddings for documents in batches
        print("\nGenerating embeddings in batches of 50")
        embeddings = []
        for step in range(0, len(documents), 50):
            docs = documents[step:step+50]
            print(f"Processing batch {step//50 + 1}/{(len(documents)-1)//50 + 1}")
            embeds = self._client.models.embed_content(
                model="models/text-embedding-004",
                contents=[doc["page_content"] for doc in docs]
            )
            vectors = [emb.values for emb in embeds.embeddings]
            embeddings.extend(vectors)
        
        print(f"\nGenerated {len(embeddings)} embeddings")
        
        # Update retriever with new documents and vectors
        self._docs.extend(documents)
        self._arr = np.vstack([self._arr, np.array(embeddings)])
        print(f"Updated vector store shape: {self._arr.shape}")

retriever = VectorStoreRetriever.from_docs(docs, client)

# Insert PDFs from Policy folder
policy_folder = "Policy"
retriever.insert_pdfs_to_vectorstore(policy_folder)
print("\nFinal retriever stats:")
print(f"Total documents: {len(retriever._docs)}")
print(f"Vector store shape: {retriever._arr.shape}")


# if __name__ == "__main__":
    # print(lookup_policy("How can I get the refund?"))
    