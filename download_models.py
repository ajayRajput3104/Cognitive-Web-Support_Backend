"""
Pre-download ML models during build phase
This prevents first-run delays and ensures models are cached
"""

from sentence_transformers import SentenceTransformer
import sys

# Model to download (must match config.py)
MODEL_NAME = 'all-MiniLM-L6-v2'

def main():
    print("=" * 80)
    print("PRE-DOWNLOADING ML MODELS")
    print("=" * 80)
    print(f"\n📥 Downloading model: {MODEL_NAME}")
    print("This will take a few minutes...")
    
    try:
        # Download and cache the model
        model = SentenceTransformer(MODEL_NAME)
        print(f"✅ Model downloaded successfully: {MODEL_NAME}")
        print(f"   Dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   Max sequence length: {model.get_max_seq_length()}")
        
        # Test encoding
        test_text = "This is a test sentence"
        embedding = model.encode([test_text])
        print(f"✅ Model test successful (embedding shape: {embedding.shape})")
        
        print("\n" + "=" * 80)
        print("MODEL DOWNLOAD COMPLETE")
        print("=" * 80 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to download model: {e}")
        print("Please check your internet connection and try again\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())