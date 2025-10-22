"""
Vector Store - MEMORY OPTIMIZED
Persistent Vector Database with Pinecone + Lazy Loading + Aggressive Cleanup
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict
import gc
import torch

# NEW Pinecone API (2024)
from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    MEMORY OPTIMIZED vector database using:
    - Lazy loading (model loads only when needed)
    - Small batch processing (8 items at a time)
    - Aggressive garbage collection
    - CPU-only mode (no GPU overhead)
    """
    
    def __init__(self):
        """Initialize vector store WITHOUT loading heavy model yet"""
        try:
            logger.info("🧠 Initializing Vector Store (memory optimized)...")
            
            # DON'T load model here - use lazy loading!
            self._embedding_model = None
            self._model_loaded = False
            
            # Configure PyTorch for minimal memory
            torch.set_num_threads(1)  # Single thread to save memory
            
            # Initialize Pinecone with NEW API
            if PINECONE_API_KEY:
                # Create Pinecone client
                self.pc = Pinecone(api_key=PINECONE_API_KEY)
                
                # Check if index exists, create if not
                existing_indexes = [index.name for index in self.pc.list_indexes()]
                
                if PINECONE_INDEX_NAME not in existing_indexes:
                    logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
                    self.pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=384,  # Dimension for paraphrase-MiniLM-L3-v2
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'  # Free tier region
                        )
                    )
                    logger.info(f"✅ Created index: {PINECONE_INDEX_NAME}")
                
                # Connect to index
                self.index = self.pc.Index(PINECONE_INDEX_NAME)
                self.use_pinecone = True
                logger.info(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
            else:
                logger.warning("⚠️  Pinecone API key not found - using in-memory storage")
                self.database = defaultdict(list)
                self.use_pinecone = False
            
            logger.info("✅ Vector Store initialized (model will load on first use)")
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}", exc_info=True)
            # Fallback to in-memory
            self.database = defaultdict(list)
            self.use_pinecone = False
            self._embedding_model = None
            logger.warning("⚠️  Falling back to in-memory storage")
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """
        LAZY LOADING: Only load model when first needed
        This saves 300-400MB of startup memory!
        """
        if self._embedding_model is None:
            logger.info(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
            logger.info("   (This happens only once, on first query)")
            
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self._model_loaded = True
            
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   Dimension: {self._embedding_model.get_sentence_embedding_dimension()}")
            logger.info(f"   Max sequence: {self._embedding_model.get_max_seq_length()}")
            
            # Log memory usage
            self._log_memory()
            
        return self._embedding_model
    
    def _log_memory(self):
        """Log current memory usage"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"💾 Current memory usage: {mem_mb:.1f} MB")
        except ImportError:
            pass  # psutil not available, skip logging
    
    def ingest_chunks(self, chunks: List[Dict], domain: str):
        """
        Convert text chunks to embeddings with AGGRESSIVE memory management
        
        Args:
            chunks: List of text chunks with metadata
            domain: Domain name for indexing
        """
        if not chunks:
            logger.warning("No chunks to ingest")
            return
        
        logger.info(f"📥 Ingesting {len(chunks)} chunks for {domain}")
        logger.info(f"   Using batch size: {EMBEDDING_BATCH_SIZE}")
        
        try:
            # Extract texts
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings in SMALL batches to prevent OOM
            logger.info("🔄 Generating embeddings (this may take a moment)...")
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=EMBEDDING_BATCH_SIZE,  # Small batches (8 instead of 32)
                show_progress_bar=False,
                convert_to_numpy=True,  # Use numpy instead of torch tensors (lighter)
                device='cpu'  # Force CPU (no GPU overhead)
            )
            
            # CRITICAL: Aggressive memory cleanup after encoding
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Embeddings generated, storing in database...")
            
            if self.use_pinecone:
                # Store in Pinecone (NEW API format)
                vectors = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vector_id = f"{domain}_{i}_{hash(chunk['url']) % 10**8}"
                    
                    # Limit metadata size (Pinecone has limits)
                    text_preview = chunk['text'][:800] if len(chunk['text']) > 800 else chunk['text']
                    
                    vectors.append({
                        "id": vector_id,
                        "values": embedding.tolist(),
                        "metadata": {
                            "text": text_preview,
                            "url": chunk['url'],
                            "domain": domain
                        }
                    })
                
                # Upsert in batches (NEW API)
                batch_size = 50  # Smaller batches for memory
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i+batch_size]
                    self.index.upsert(vectors=batch)
                    
                    # Cleanup after each batch
                    gc.collect()
                
                logger.info(f"✅ Stored {len(chunks)} chunks in Pinecone")
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
                
                logger.info(f"✅ Stored {len(chunks)} chunks in memory")
            
            # Final memory cleanup
            del embeddings
            del texts
            gc.collect()
            
            self._log_memory()
                
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            # Cleanup on error
            gc.collect()
            raise
    
    def retrieve_relevant(
        self, 
        query: str, 
        domain: str, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks with memory-efficient processing
        
        Args:
            query: User query
            domain: Domain to search within
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with relevance scores
        """
        try:
            # Generate query embedding (single item, minimal memory)
            query_embedding = self.embedding_model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                device='cpu'
            )[0]
            
            # Cleanup immediately
            gc.collect()
            
            if self.use_pinecone:
                # Query Pinecone
                results = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=top_k,
                    filter={"domain": {"$eq": domain}},
                    include_metadata=True
                )
                
                # Format results
                relevant_chunks = []
                for match in results.get('matches', []):
                    relevant_chunks.append({
                        'text': match['metadata']['text'],
                        'url': match['metadata']['url'],
                        'relevance_score': float(match['score'])
                    })
                
                # Cleanup
                del query_embedding
                gc.collect()
                
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
                
                results = [{
                    'text': chunk['text'],
                    'url': chunk['url'],
                    'relevance_score': score
                } for chunk, score in chunks_with_scores[:top_k]]
                
                # Cleanup
                del query_embedding
                gc.collect()
                
                return results
                
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            gc.collect()
            return []
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a specific domain"""
        if self.use_pinecone:
            try:
                # Query with filter to check if domain exists
                stats_query = self.index.query(
                    vector=[0.0] * 384,
                    top_k=1,
                    filter={"domain": {"$eq": domain}},
                    include_metadata=False
                )
                exists = len(stats_query.get('matches', [])) > 0
                return {
                    'domain': domain,
                    'chunks': 0,  # Pinecone doesn't provide easy count
                    'exists': exists
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
        """Get list of all domains in the database"""
        if self.use_pinecone:
            return []  # Pinecone doesn't provide easy domain listing
        else:
            return list(self.database.keys())
    
    def clear_domain(self, domain: str) -> bool:
        """Clear all data for a specific domain"""
        if self.use_pinecone:
            try:
                self.index.delete(filter={"domain": {"$eq": domain}})
                logger.info(f"✅ Cleared domain from Pinecone: {domain}")
                gc.collect()
                return True
            except Exception as e:
                logger.error(f"Failed to clear domain: {e}")
                return False
        else:
            if domain in self.database:
                del self.database[domain]
                gc.collect()
                return True
            return False
    
    def get_total_chunks(self) -> int:
        """Get total number of chunks across all domains"""
        if self.use_pinecone:
            try:
                stats = self.index.describe_index_stats()
                return stats.get('total_vector_count', 0)
            except:
                return 0
        else:
            return sum(len(chunks) for chunks in self.database.values())
    
    def health_check(self) -> bool:
        """Check if vector store is healthy"""
        try:
            if self.use_pinecone:
                self.index.describe_index_stats()
            return True
        except:
            return False
    
    def cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("🧹 Cleaning up vector store resources...")
        
        # Clear model from memory
        if self._embedding_model is not None:
            del self._embedding_model
            self._embedding_model = None
            self._model_loaded = False
        
        # Aggressive cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Vector store cleanup complete")