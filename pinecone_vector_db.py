"""
Pinecone Vector Database Adapter for SYSK RAG Chatbot
Drop-in replacement for ChromaDB that persists across Streamlit reboots
"""

import os
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import json


class PineconeVectorDatabase:
    """Pinecone vector database for transcript storage and retrieval"""
    
    def __init__(self, 
                 index_name: str = "sysk-transcripts",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 dimension: int = 384):
        """
        Initialize Pinecone vector database
        
        Args:
            index_name: Name of Pinecone index
            embedding_model_name: Sentence transformer model name
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        # Get API credentials from environment or Streamlit secrets
        try:
            import streamlit as st
            api_key = st.secrets.get("PINECONE_API_KEY")
        except:
            api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in Streamlit secrets or environment")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Create or connect to index
        self._initialize_index()
        
        # Get index object
        self.index = self.pc.Index(self.index_name)
    
    def _initialize_index(self):
        """Create index if it doesn't exist"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            # Create new index with serverless spec (free tier)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
    
    def add(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Add documents to Pinecone index
        
        Args:
            documents: List of text documents
            metadatas: List of metadata dicts
            ids: List of unique IDs
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Prepare vectors for upsert
        vectors = []
        for i, (doc_id, embedding, metadata, text) in enumerate(zip(ids, embeddings, metadatas, documents)):
            # Store text in metadata (Pinecone doesn't store it separately)
            metadata_with_text = {
                **metadata,
                'text': text[:1000]  # Store first 1000 chars for retrieval
            }
            
            vectors.append({
                'id': doc_id,
                'values': embedding,
                'metadata': metadata_with_text
            })
        
        # Upsert to Pinecone (batch size 100)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def query(self, 
              query_texts: List[str], 
              n_results: int = 10,
              filter: Optional[Dict] = None) -> Dict:
        """
        Query Pinecone index (mimics ChromaDB interface)
        
        Args:
            query_texts: List of query strings
            n_results: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            Dict with ChromaDB-compatible format
        """
        if not query_texts:
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        # Get embedding for first query (single query support)
        query_embedding = self.embedding_model.encode(query_texts[0]).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True,
            filter=filter
        )
        
        # Convert to ChromaDB format
        ids = []
        documents = []
        metadatas = []
        distances = []
        
        for match in results['matches']:
            ids.append(match['id'])
            metadata = match['metadata']
            documents.append(metadata.pop('text', ''))  # Extract text from metadata
            metadatas.append(metadata)
            # Convert similarity score to distance (1 - score)
            distances.append(1 - match['score'])
        
        return {
            'ids': [ids],
            'documents': [documents],
            'metadatas': [metadatas],
            'distances': [distances]
        }
    
    def get(self, ids: Optional[List[str]] = None) -> Dict:
        """
        Get all vectors or specific vectors by ID
        
        Args:
            ids: Optional list of IDs to fetch
            
        Returns:
            Dict with ids, metadatas, documents
        """
        if ids:
            # Fetch specific vectors
            try:
                results = self.index.fetch(ids=ids)
                
                fetched_ids = []
                metadatas = []
                documents = []
                
                for vector_id, vector_data in results.get('vectors', {}).items():
                    fetched_ids.append(vector_id)
                    metadata = vector_data.get('metadata', {}).copy()
                    documents.append(metadata.pop('text', ''))
                    metadatas.append(metadata)
                
                return {
                    'ids': fetched_ids,
                    'metadatas': metadatas,
                    'documents': documents
                }
            except Exception as e:
                # Return empty if fetch fails
                return {
                    'ids': [],
                    'metadatas': [],
                    'documents': []
                }
        else:
            # Get all vectors using list/pagination
            try:
                # Check if index has any vectors first
                stats = self.index.describe_index_stats()
                if stats.get('total_vector_count', 0) == 0:
                    return {
                        'ids': [],
                        'metadatas': [],
                        'documents': []
                    }
                
                all_ids = []
                
                # List all vector IDs (paginated)
                for ids_batch in self.index.list(namespace=''):
                    all_ids.extend(ids_batch)
                
                if not all_ids:
                    return {
                        'ids': [],
                        'metadatas': [],
                        'documents': []
                    }
                
                # Fetch all vectors in batches
                all_metadatas = []
                all_documents = []
                batch_size = 1000
                
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    results = self.index.fetch(ids=batch_ids)
                    
                    for vector_id in batch_ids:
                        if vector_id in results.get('vectors', {}):
                            vector_data = results['vectors'][vector_id]
                            metadata = vector_data.get('metadata', {}).copy()
                            all_documents.append(metadata.pop('text', ''))
                            all_metadatas.append(metadata)
                
                return {
                    'ids': all_ids,
                    'metadatas': all_metadatas,
                    'documents': all_documents
                }
            except Exception as e:
                # Return empty if anything fails
                return {
                    'ids': [],
                    'metadatas': [],
                    'documents': []
                }
    
    def delete(self, ids: Optional[List[str]] = None, delete_all: bool = False):
        """
        Delete vectors by ID or delete all
        
        Args:
            ids: List of IDs to delete (optional)
            delete_all: If True, delete all vectors in index
        """
        if delete_all:
            # Delete all vectors from index
            self.index.delete(delete_all=True)
        elif ids:
            # Delete specific vectors in batches
            batch_size = 1000
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i + batch_size]
                self.index.delete(ids=batch)
        else:
            # No-op if neither specified
            pass
    
    def count(self) -> int:
        """Get total number of vectors in index"""
        stats = self.index.describe_index_stats()
        return stats['total_vector_count']
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                return {
                    'total_chunks': 0,
                    'total_episodes': 0,
                    'full_episodes': 0,
                    'short_stuff': 0
                }
            
            # Sample vectors to get episode counts
            # Note: Pinecone doesn't allow fetching all metadata easily
            # We'll use a query to sample and estimate
            try:
                # Query with a dummy vector to get some results
                import numpy as np
                dummy_vector = np.zeros(self.dimension).tolist()
                
                # Get a sample of 1000 vectors
                sample_results = self.index.query(
                    vector=dummy_vector,
                    top_k=min(1000, total_vectors),
                    include_metadata=True
                )
                
                # Count unique episodes and types from sample
                unique_episodes = set()
                full_episodes = 0
                short_stuff = 0
                
                for match in sample_results.get('matches', []):
                    metadata = match.get('metadata', {})
                    filename = metadata.get('filename', '')
                    if filename:
                        unique_episodes.add(filename)
                    
                    episode_type = metadata.get('episode_type', '')
                    if episode_type == 'Full Episode':
                        full_episodes += 1
                    elif episode_type == 'Short Stuff':
                        short_stuff += 1
                
                return {
                    'total_chunks': total_vectors,
                    'total_episodes': len(unique_episodes),
                    'full_episodes': full_episodes,
                    'short_stuff': short_stuff
                }
            except:
                # If sampling fails, return basic stats
                return {
                    'total_chunks': total_vectors,
                    'total_episodes': 0,
                    'full_episodes': 0,
                    'short_stuff': 0
                }
        except Exception as e:
            return {
                'total_chunks': 0,
                'total_episodes': 0,
                'full_episodes': 0,
                'short_stuff': 0
            }


# Wrapper class to maintain ChromaDB-like interface
class VectorDatabase:
    """Wrapper to maintain existing interface while using Pinecone"""
    
    def __init__(self, collection_name: str = "sysk-transcripts", persist_directory: str = None):
        """
        Initialize vector database (uses Pinecone instead of ChromaDB)
        
        Args:
            collection_name: Name for Pinecone index
            persist_directory: Ignored (Pinecone is cloud-based)
        """
        # Initialize Pinecone database
        self.collection = PineconeVectorDatabase(index_name=collection_name)
        
        # For compatibility with existing code
        self.persist_directory = "pinecone"  # Symbolic
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return self.collection.get_stats()
