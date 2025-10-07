# to reset the dataload on the particular pinecone index and then reload the update on the pinecone (to remove data redundancy)
 
import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "travel-sprl-mi"

def reset_pinecone_index():
    """Delete all vectors from Pinecone index"""
    print("🗑️  Resetting Pinecone index...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, if not create it
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"⚠️  Index '{INDEX_NAME}' not found. Creating new index...")
        from pinecone import ServerlessSpec
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created new index: {INDEX_NAME}")
    
    index = pc.Index(INDEX_NAME)
    
    # Wait for index to be ready
    print("⏳ Waiting for index to be ready...")
    time.sleep(10)
    
    # Get current stats
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    
    if total_vectors == 0:
        print("✅ Index is already empty!")
        return index
    
    print(f"📊 Current vectors in index: {total_vectors}")
    
    # Delete all vectors (ye sab vectors ko delete kr dega)
    index.delete(delete_all=True)
    
    print("✅ Successfully deleted all vectors from index!")
    
    # Verify deletion
    stats = index.describe_index_stats()
    print(f"📊 Vectors after deletion: {stats.total_vector_count}")
    
    return index

def reload_travel_records(index, json_path="travel_records.json"):
    """Reload travel records to Pinecone"""
    print(f"📤 Reloading travel records from {json_path}...")
    
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found!")
        return False
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load JSON data
    with open(json_path, "r") as f:
        travel_data = json.load(f)
    
    records = travel_data.get("Travel Request", [])
    
    if not records:
        print("⚠️  No travel records found in JSON!")
        return False
    
    print(f"📝 Found {len(records)} records to upload")
    
    vectors = []
    for i, record in enumerate(records):
        def safe_str(value):
            return str(value).strip() if value is not None else ""
        
        employee_original = safe_str(record.get("employee", "")).strip()
        if not employee_original:
            continue
        
        employee_normalized = employee_original.lower()
        
        text = (
            f"Travel Request:\n"
            f"Employee: {employee_original}\n"
            f"Department: {safe_str(record.get('department'))}\n"
            f"From: {safe_str(record.get('origin'))} to {safe_str(record.get('destination'))}\n"
            f"Date: {safe_str(record.get('date'))}\n"
            f"Email: {safe_str(record.get('email'))}\n"
            f"Cost: {safe_str(record.get('cost'))}"
        )
        
        metadata = {
            "source": "travel_json",
            "type": "travel",
            "employee": employee_normalized,
            "employee_original": employee_original,
            "origin": safe_str(record.get("origin")),
            "department": safe_str(record.get("department")),
            "destination": safe_str(record.get("destination")),
            "date": safe_str(record.get("date")),
            "email": safe_str(record.get("email")),
            "cost": float(record.get("cost", 0)) if record.get("cost") else 0,
            "text": text
        }
        
        embedding = embedding_model.embed_query(text)
        vector_id = f"travel-{i}"
        vectors.append((vector_id, embedding, metadata))
    
    # Upload in batches
    if vectors:
        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc="📤 Uploading"):
            batch = vectors[i:i + batch_size]
            index.upsert(batch)
        
        print(f"✅ Successfully uploaded {len(vectors)} records!")
        
        # Show sample
        print(f"📋 Sample uploaded records:")
        for i, (vid, emb, meta) in enumerate(vectors[:3]):
            print(f"   {i+1}. {meta['employee_original']} | {meta['department']}")
        
        return True
    
    return False

def remove_flag_files():
    """Remove flag files to allow fresh upload"""
    flag_files = [
        "upload_travel_records_done.flag",
        "all_data_uploaded.flag"
    ]
    
    for flag_file in flag_files:
        if os.path.exists(flag_file):
            os.remove(flag_file)
            print(f"🗑️  Removed: {flag_file}")

def main():
    """Main function - Reset and Reload"""
    print("=" * 50)
    print("🔄 PINECONE RESET & RELOAD UTILITY")
    print("=" * 50)
    
    try:
        # Step 1: Remove flag files
        print("\n[Step 1] Removing flag files...")
        remove_flag_files()
        
        # Step 2: Reset index (delete all data)
        print("\n[Step 2] Resetting Pinecone index...")
        index = reset_pinecone_index()
        
        # Step 3: Reload data
        print("\n[Step 3] Reloading travel records...")
        success = reload_travel_records(index)
        
        # Step 4: Show final stats
        print("\n[Step 4] Final Statistics:")
        stats = index.describe_index_stats()
        print(f"📊 Total vectors in index: {stats.total_vector_count}")
        
        if success:
            print("\n✅ Reset and reload completed successfully!")
        else:
            print("\n⚠️  Reset completed but reload had issues!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    confirm = input("⚠️  This will DELETE all data and reload. Continue? (yes/no): ")
    if confirm.lower() == "yes":
        main()
    else:
        print("❌ Operation cancelled!")