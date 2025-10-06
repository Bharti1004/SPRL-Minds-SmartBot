import os
import json
import sqlite3
import pickle
import datetime
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def resolve_coreferences(context, question):
    doc = nlp(context + " " + question)
    return doc.coref_resolved if doc.has_coref else question

# ----------------- Load .env ------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env file.")
if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env file.")

# ----------------- Gemini Setup ------------------
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL_NAME = "models/gemini-2.0-flash"

# ----------------- Token Handling ------------------
def count_tokens_approx(text: str) -> int:
    return len(text) // 4

def log_token_usage(query: str, response: str):
    tokens_input = count_tokens_approx(query)
    tokens_output = count_tokens_approx(response)
    total_tokens = tokens_input + tokens_output
    log_entry = {
        "query": query,
        "response": response,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "total_tokens": total_tokens
    }
    with open("token_usage_log.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"🧒 Token usage - In: {tokens_input}, Out: {tokens_output}, Total: {total_tokens}")
    return tokens_input, tokens_output

# ----------------- SQLite DB ------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS user_tokens (user_id TEXT PRIMARY KEY, token_limit INT, token_used INT, token_remaining INT)")
    c.execute("INSERT OR IGNORE INTO user_tokens VALUES (?, ?, ?, ?)", ("user1", 10000, 0, 10000))
    
    c.execute("""CREATE TABLE IF NOT EXISTS history (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        query TEXT,
        response TEXT,
        input_tokens INT,
        output_tokens INT,
        total_tokens INT,
        timestamp TEXT
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        user_id TEXT,
        original_query TEXT,
        original_response TEXT,
        corrected_answer TEXT
    )""")
    c.execute("CREATE TABLE IF NOT EXISTS user_context (user_id TEXT PRIMARY KEY, last_person TEXT)")
    c.execute("INSERT OR IGNORE INTO user_context VALUES (?, ?)", ("user1", None))

    conn.commit()
    conn.close()

# ----------------- Pinecone Setup ------------------
def setup_pinecone():
    """Setup Pinecone connection and index"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "travel-sprl-mi"

    if index_name not in pc.list_indexes().names():
        print(f"🆕 Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
        print(f"✅ Successfully created index: {index_name}")
    else:
        print(f"✅ Using existing Pinecone index: {index_name}")

    index = pc.Index(index_name)
    return index, index_name

def upload_travel_records_to_pinecone(index, json_path="travel_records.json", flag_file="upload_travel_records_done.flag"):
    """Upload travel records to Pinecone with better name handling"""
    
    if os.path.exists(flag_file):
        print("✅ Travel records already uploaded to Pinecone. Skipping re-upload.")
        return True
    
    if not os.path.exists(json_path):
        print(f"⚠ Warning: {json_path} not found. Skipping travel records upload.")
        with open(flag_file, "w") as f:
            f.write("travel records upload skipped - file not found")
        return False

    print(f"🔄 Starting travel records upload to Pinecone...")

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with open(json_path, "r") as f:
        travel_data = json.load(f)

    records = travel_data.get("Travel Request", [])
    if not records:
        print("⚠ No travel records found.")
        with open(flag_file, "w") as f:
            f.write("travel records upload complete - no records found")
        return False

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
            "employee": employee_normalized,  # Lowercase for filtering
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

    if vectors:
        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc="📤 Uploading travel records"):
            batch = vectors[i:i + batch_size]
            index.upsert(batch)

        print(f"✅ Uploaded {len(vectors)} travel records to Pinecone.")
        
        # Show sample of records
        print(f"📝 Sample records uploaded:")
        for i, (vid, emb, meta) in enumerate(vectors[:3]):
            print(f"   {i+1}. Employee: {meta['employee_original']} | Department: {meta['department']}")
    
    with open(flag_file, "w") as f:
        f.write("travel records upload complete")
    print(f"🚩 Created flag file: {flag_file}")
    return True

def check_pinecone_stats(index):
    """Check current stats of Pinecone index"""
    try:
        stats = index.describe_index_stats()
        print(f"📊 Pinecone Index Stats:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Index fullness: {stats.index_fullness}")
        if hasattr(stats, 'namespaces') and stats.namespaces:
            for namespace, info in stats.namespaces.items():
                print(f"   Namespace '{namespace}': {info.vector_count} vectors")
    except Exception as e:
        print(f"⚠ Could not fetch Pinecone stats: {e}")

# ----------------- Force Upload Function ------------------
def force_reupload_travel_records():
    """Force re-upload of travel records by removing flag files"""
    flag_files = [
        "upload_travel_records_done.flag",
        "all_data_uploaded.flag"
    ]
    
    for flag_file in flag_files:
        if os.path.exists(flag_file):
            os.remove(flag_file)
            print(f"🗑 Removed flag file: {flag_file}")

# ----------------- Main Function - Load Once and Stop ------------------
def main():
    """Main function that loads data once and stops"""
    print("🚀 Starting Pinecone data loader...")
    
    try:
        # Initialize database
        print("📊 Initializing database...")
        init_db()
        
        # Setup Pinecone
        print("🔗 Setting up Pinecone connection...")
        index, index_name = setup_pinecone()
        
        # Check current stats
        print("📈 Checking current Pinecone stats...")
        check_pinecone_stats(index)
        
        # Check if index is empty but flag exists - force reupload
        stats = index.describe_index_stats()
        if stats.total_vector_count == 0 and os.path.exists("upload_travel_records_done.flag"):
            print("🔄 Index is empty but flag exists. Forcing re-upload...")
            force_reupload_travel_records()
        
        # Upload travel records
        print("📤 Uploading travel records...")
        success = upload_travel_records_to_pinecone(index)
        
        if success:
            print("✅ Data upload completed successfully!")
            # Check final stats
            print("📈 Final Pinecone stats:")
            check_pinecone_stats(index)
        else:
            print("⚠ Data upload completed with warnings.")
            
        print("🏁 Process completed. Exiting...")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    # Check for force upload argument
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        print("🔄 Force upload requested...")
        force_reupload_travel_records()
    
    main()