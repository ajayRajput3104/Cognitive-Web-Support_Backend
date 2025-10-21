"""
Pre-download ML models during build phase - WITH ERROR HANDLING
This prevents first-run delays and ensures models are cached
"""

import sys
import os

# Model to download (must match config.py)
MODEL_NAME = 'all-MiniLM-L6-v2'

def main():
    print("=" * 80)
    print("PRE-DOWNLOADING ML MODELS")
    print("=" * 80)
    print(f"\n📥 Downloading model: {MODEL_NAME}")
    print("This will take a few minutes...")
    
    try:
        # Set cache directory
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Import with error handling
        print("\n1️⃣ Importing sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        
        # Download and cache the model
        print(f"2️⃣ Downloading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        
        print(f"✅ Model downloaded successfully: {MODEL_NAME}")
        print(f"   Dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   Max sequence length: {model.get_max_seq_length()}")
        
        # Test encoding
        print("3️⃣ Testing model...")
        test_text = "This is a test sentence"
        embedding = model.encode([test_text])
        print(f"✅ Model test successful (embedding shape: {embedding.shape})")
        
        print("\n" + "=" * 80)
        print("MODEL DOWNLOAD COMPLETE")
        print("=" * 80 + "\n")
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