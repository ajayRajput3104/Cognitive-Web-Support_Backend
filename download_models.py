"""
Pre-download LIGHTWEIGHT ML model during build phase
This prevents first-run delays and ensures models are cached
OPTIMIZED
"""

import sys
import os

# LIGHTWEIGHT model (33MB vs 420MB!)
MODEL_NAME = 'paraphrase-MiniLM-L3-v2'

def main():
    print("=" * 80)
    print("PRE-DOWNLOADING LIGHTWEIGHT ML MODEL (MEMORY OPTIMIZED)")
    print("=" * 80)
    print(f"\n📥 Downloading model: {MODEL_NAME}")
    print("This is a lightweight model (33MB) optimized for 512MB RAM limit")
    print("This will take ~30 seconds...\n")
    
    try:
        # Set cache directory
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✅ Cache directory: {cache_dir}")
        
        # Import with error handling
        print("\n1️⃣  Importing sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        
        # Download and cache the model
        print(f"2️⃣  Downloading lightweight model: {MODEL_NAME}")
        print("   (This is 92% smaller than the default model!)")
        
        model = SentenceTransformer(MODEL_NAME)
        
        print(f"\n✅ Model downloaded successfully!")
        print(f"   Name: {MODEL_NAME}")
        print(f"   Dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   Max sequence length: {model.get_max_seq_length()}")
        
        # Test encoding with minimal memory
        print("\n3️⃣  Testing model...")
        test_text = "This is a test sentence"
        embedding = model.encode([test_text], show_progress_bar=False)
        print(f"✅ Model test successful (embedding shape: {embedding.shape})")
        
        # Cleanup
        del model
        del embedding
        import gc
        gc.collect()
        
        print("\n" + "=" * 80)
        print("MODEL DOWNLOAD COMPLETE - MEMORY OPTIMIZED")
        print("=" * 80)
        print("\n💡 Tips:")
        print("   - This lightweight model uses only 33MB vs 420MB")
        print("   - Model will load lazily on first API request")
        print("   - Expected startup memory: ~120MB (vs 450MB before)")
        print("   - Peak memory during query: ~280MB (vs 550MB+ before)")
        print("\n")
        return 0
        
    except ImportError as e:
        print(f"\n⚠️  WARNING: Import error - {e}")
        print("Models will be downloaded on first application run")
        print("This is not critical - application will still work\n")
        return 0  # Don't fail build
        
    except Exception as e:
        print(f"\n⚠️  WARNING: Failed to download model - {e}")
        print("Models will be downloaded on first application run")
        print("This is not critical - application will still work\n")
        return 0  # Don't fail build

if __name__ == "__main__":
    sys.exit(main())