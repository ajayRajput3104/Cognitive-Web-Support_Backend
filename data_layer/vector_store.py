"""
Vector Store - Persistent Vector Database with Pinecone
Manages embeddings and semantic search with permanent storage
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import pinecone
from collections import defaultdict

from config import (
    PINECONE_API_KEY, 
    PINECONE_ENVIRONMENT, 
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Persistent vector database using Pinecone
    Stores document embeddings permanently across server restarts
    """
    
    def __init__(self):
        """Initialize vector store with Pinecone and embedding model"""
        try:
            logger.info("Initializing Vector Store...")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"✓ Loaded embedding model: {EMBEDDING_MODEL}")
            
            # Initialize Pinecone
            if PINECONE_API_KEY:
                pinecone.init(
                    api_key=PINECONE_API_KEY,
                    environment=PINECONE_ENVIRONMENT
                )
                
                # Get or create index
                if PINECONE_INDEX_NAME not in pinecone.list_indexes():
                    logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
                    pinecone.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=384,  # Dimension for all-MiniLM-L6-v2
                        metric="cosine"
                    )
                
                self.index = pinecone.Index(PINECONE_INDEX_NAME)
                self.use_pinecone = True
                logger.info(f"✓ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
            else:
                logger.warning("⚠ Pinecone API key not found - using in-memory storage")
                self.database = defaultdict(list)
                self.use_pinecone = False
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}", exc_info=True)
            # Fallback to in-memory
            self.database = defaultdict(list)
            self.use_pinecone = False
            logger.warning("⚠ Falling back to in-memory storage")
    
    def ingest_chunks(self, chunks: List[Dict], domain: str):
        """
        Convert text chunks to embeddings and store them
        
        Args:
            chunks: List of text chunks with metadata
            domain: Domain name for indexing
        """
        if not chunks:
            logger.warning("No chunks to ingest")
            return
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        try:
            # Extract texts
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=32, 
                show_progress_bar=False
            )
            
            if self.use_pinecone:
                # Store in Pinecone
                vectors = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vector_id = f"{domain}_{i}_{hash(chunk['url'])}"
                    vectors.append((
                        vector_id,
                        embedding.tolist(),
                        {
                            "text": chunk['text'][:1000],  # Pinecone metadata limit
                            "url": chunk['url'],
                            "domain": domain
                        }
                    ))
                
                # Upsert in batches
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i+batch_size]
                    self.index.upsert(vectors=batch)
                
                logger.info(f"✓ Stored {len(chunks)} chunks in Pinecone")
            else:
                # Store in memory (fallback)
                if domain in self.database:
                    self.database[domain].clear()
                
                for chunk, embedding in zip(chunks, embeddings):
                    self.database[domain].append({
                        'text': chunk['text'],
                        'url': chunk['url'],
                        'embedding': embedding.tolist()
                    })
                
                logger.info(f"✓ Stored {len(chunks)} chunks in memory")
                
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise
    
    def retrieve_relevant(
        self, 
        query: str, 
        domain: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks using cosine similarity
        
        Args:
            query: User query
            domain: Domain to search within
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with relevance scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            if self.use_pinecone:
                # Query Pinecone
                results = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=top_k,
                    filter={"domain": domain},
                    include_metadata=True
                )
                
                # Format results
                relevant_chunks = []
                for match in results['matches']:
                    relevant_chunks.append({
                        'text': match['metadata']['text'],
                        'url': match['metadata']['url'],
                        'relevance_score': float(match['score'])
                    })
                
                return relevant_chunks
            else:
                # Search in memory (fallback)
                if domain not in self.database or not self.database[domain]:
                    return []
                
                chunks_with_scores = []
                for chunk in self.database[domain]:
                    chunk_embedding = np.array(chunk['embedding'])
                    
                    # Cosine similarity
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    
                    chunks_with_scores.append((chunk, float(similarity)))
                
                # Sort and return top K
                chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                return [{
                    'text': chunk['text'],
                    'url': chunk['url'],
                    'relevance_score': score
                } for chunk, score in chunks_with_scores[:top_k]]
                
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return []
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """
        Get statistics for a specific domain
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary with domain statistics
        """
        if self.use_pinecone:
            try:
                # Query to count domain vectors
                stats = self.index.describe_index_stats()
                # Note: Pinecone doesn't provide per-namespace stats easily
                # This is a limitation of the free tier
                return {
                    'domain': domain,
                    'chunks': 0,  # Would need custom tracking
                    'exists': True
                }
            except:
                return {'domain': domain, 'chunks': 0, 'exists': False}
        else:
            if domain not in self.database:
                return {'domain': domain, 'chunks': 0, 'exists': False}
            return {
                'domain': domain,
                'chunks': len(self.database[domain]),
                'exists': True
            }
    
    def get_all_domains(self) -> List[str]:
        """
        Get list of all domains in the database
        
        Returns:
            List of domain names
        """
        if self.use_pinecone:
            # Pinecone doesn't provide easy domain listing
            # Would need custom tracking
            return []
        else:
            return list(self.database.keys())
    
    def clear_domain(self, domain: str) -> bool:
        """
        Clear all data for a specific domain
        
        Args:
            domain: Domain to clear
            
        Returns:
            True if domain was cleared, False if didn't exist
        """
        if self.use_pinecone:
            try:
                # Delete by metadata filter
                self.index.delete(filter={"domain": domain})
                logger.info(f"✓ Cleared domain from Pinecone: {domain}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear domain: {e}")
                return False
        else:
            if domain in self.database:
                del self.database[domain]
                return True
            return False
    
    def get_total_chunks(self) -> int:
        """
        Get total number of chunks across all domains
        
        Returns:
            Total chunk count
        """
        if self.use_pinecone:
            try:
                stats = self.index.describe_index_stats()
                return stats.get('total_vector_count', 0)
            except:
                return 0
        else:
            return sum(len(chunks) for chunks in self.database.values())
    
    def health_check(self) -> bool:
        """
        Check if vector store is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if self.use_pinecone:
                self.index.describe_index_stats()
            return True
        except:
            return False
    
    def cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("Cleaning up vector store resources...")
        # Nothing to cleanup for Pinecone (persistent)