# Advanced json + (append more than one json file or you may add new data to existing json) -> pinecone with 
#                                                                                               incremental uploads

import os
import json
import sqlite3
import datetime
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import hashlib

# ----------------- Load .env ------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env file.")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file.")

# ----------------- OpenAI Setup ------------------
client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL_NAME = "gpt-4o-mini"

# ----------------- SQLite DB ------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    # Existing tables
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
    
    # NEW: Upload tracking table
    c.execute("""CREATE TABLE IF NOT EXISTS upload_history (
        record_hash TEXT PRIMARY KEY,
        employee TEXT,
        date TEXT,
        source_file TEXT,
        upload_timestamp TEXT,
        vector_id TEXT
    )""")
    
    # NEW: File tracking table
    c.execute("""CREATE TABLE IF NOT EXISTS processed_files (
        file_name TEXT PRIMARY KEY,
        file_path TEXT,
        records_count INT,
        upload_timestamp TEXT,
        status TEXT
    )""")

    conn.commit()
    conn.close()

# ----------------- Pinecone Setup ------------------
def setup_pinecone():
    """Setup Pinecone connection and index"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "travel-sprl"

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

# ----------------- Hash Generation for Duplicate Detection ------------------
def generate_record_hash(employee, date, origin, destination):
    """Generate unique hash for a travel record"""
    unique_string = f"{employee.lower().strip()}|{date}|{origin}|{destination}"
    return hashlib.md5(unique_string.encode()).hexdigest()

# ----------------- Check if Record Already Uploaded ------------------
def is_record_uploaded(record_hash):
    """Check if record already exists in database"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT record_hash FROM upload_history WHERE record_hash = ?", (record_hash,))
    result = c.fetchone()
    conn.close()
    return result is not None

# ----------------- Save Upload History ------------------
def save_upload_history(record_hash, employee, date, source_file, vector_id):
    """Save upload history to database"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute("""INSERT OR IGNORE INTO upload_history 
                 (record_hash, employee, date, source_file, upload_timestamp, vector_id)
                 VALUES (?, ?, ?, ?, ?, ?)""",
              (record_hash, employee, date, source_file, timestamp, vector_id))
    conn.commit()
    conn.close()

# ----------------- Save File Processing Status ------------------
def save_file_status(file_name, file_path, records_count, status="completed"):
    """Save file processing status"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute("""INSERT OR REPLACE INTO processed_files 
                 (file_name, file_path, records_count, upload_timestamp, status)
                 VALUES (?, ?, ?, ?, ?)""",
              (file_name, file_path, records_count, timestamp, status))
    conn.commit()
    conn.close()

# ----------------- Get Processed Files ------------------
def get_processed_files():
    """Get list of already processed files"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT file_name, records_count, upload_timestamp FROM processed_files WHERE status='completed'")
    results = c.fetchall()
    conn.close()
    return results

# ----------------- Find JSON Files ------------------
def find_json_files(directory="."):
    """Find all JSON files in directory"""
    json_files = []
    for file in os.listdir(directory):
        if file.endswith(".json") and "token_usage" not in file.lower():
            json_files.append(file)
    return sorted(json_files)

# ----------------- Upload Travel Records (Advanced) ------------------
def upload_travel_records_advanced(index, json_files=None):
    """
    Advanced upload with:
    - Multiple file support
    - Duplicate detection
    - Incremental uploads
    - User interaction
    """
    
    print("\n" + "="*60)
    print("🚀 ADVANCED TRAVEL RECORDS UPLOADER")
    print("="*60 + "\n")
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # If no files specified, find all JSON files
    if json_files is None:
        json_files = find_json_files()
    
    if not json_files:
        print("⚠️ No JSON files found in current directory!")
        return False
    
    # Show available files
    print("📂 Available JSON files:")
    for i, file in enumerate(json_files, 1):
        size = os.path.getsize(file) / 1024  # KB
        print(f"   {i}. {file} ({size:.2f} KB)")
    
    # Show already processed files
    processed = get_processed_files()
    if processed:
        print("\n✅ Already processed files:")
        for fname, count, timestamp in processed:
            print(f"   - {fname}: {count} records (uploaded on {timestamp[:10]})")
    
    # Ask user which files to process
    print("\n" + "-"*60)
    user_input = input("📝 Enter file numbers to upload (comma-separated) or 'all': ").strip()
    
    if user_input.lower() == 'all':
        selected_files = json_files
    else:
        try:
            indices = [int(x.strip()) - 1 for x in user_input.split(",")]
            selected_files = [json_files[i] for i in indices if 0 <= i < len(json_files)]
        except:
            print("❌ Invalid input! Exiting...")
            return False
    
    if not selected_files:
        print("⚠️ No files selected!")
        return False
    
    print(f"\n📤 Selected {len(selected_files)} file(s) for upload:")
    for f in selected_files:
        print(f"   - {f}")
    
    # Confirm before proceeding
    confirm = input("\n⚠️ Proceed with upload? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("❌ Upload cancelled by user.")
        return False
    
    # Process each file
    total_uploaded = 0
    total_skipped = 0
    
    for json_file in selected_files:
        print(f"\n{'='*60}")
        print(f"📄 Processing: {json_file}")
        print(f"{'='*60}")
        
        if not os.path.exists(json_file):
            print(f"⚠️ File not found: {json_file}. Skipping...")
            continue
        
        # Load JSON data
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                travel_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ Error reading JSON: {e}")
                save_file_status(json_file, json_file, 0, "error")
                continue
        
        records = travel_data.get("Travel Request", [])
        if not records:
            print(f"⚠️ No travel records found in {json_file}")
            save_file_status(json_file, json_file, 0, "empty")
            continue
        
        print(f"📊 Found {len(records)} records in file")
        
        # Process records
        vectors = []
        new_count = 0
        skipped_count = 0
        
        for i, record in enumerate(tqdm(records, desc="🔍 Checking records")):
            def safe_str(value):
                return str(value).strip() if value is not None else ""
            
            employee_original = safe_str(record.get("employee", "")).strip()
            if not employee_original:
                continue
            
            employee_normalized = employee_original.lower()
            date = safe_str(record.get("date"))
            origin = safe_str(record.get("origin"))
            destination = safe_str(record.get("destination"))
            
            # Generate hash for duplicate detection
            record_hash = generate_record_hash(employee_original, date, origin, destination)
            
            # Check if already uploaded
            if is_record_uploaded(record_hash):
                skipped_count += 1
                continue
            
            # Create embedding text
            text = (
                f"Travel Request:\n"
                f"Employee: {employee_original}\n"
                f"Department: {safe_str(record.get('department'))}\n"
                f"From: {origin} to {destination}\n"
                f"Date: {date}\n"
                f"Email: {safe_str(record.get('email'))}\n"
                f"Cost: {safe_str(record.get('cost'))}"
            )
            
            metadata = {
                "source": json_file,
                "type": "travel",
                "employee": employee_normalized,
                "employee_original": employee_original,
                "origin": origin,
                "department": safe_str(record.get("department")),
                "destination": destination,
                "date": date,
                "email": safe_str(record.get("email")),
                "cost": float(record.get("cost", 0)) if record.get("cost") else 0,
                "text": text,
                "record_hash": record_hash
            }
            
            # Generate embedding
            embedding = embedding_model.embed_query(text)
            vector_id = f"travel-{uuid4().hex[:12]}"
            vectors.append((vector_id, embedding, metadata, record_hash))
            new_count += 1
        
        # Upload to Pinecone
        if vectors:
            batch_size = 100
            print(f"\n📤 Uploading {len(vectors)} new records to Pinecone...")
            
            for i in tqdm(range(0, len(vectors), batch_size), desc="⬆️ Uploading batches"):
                batch = vectors[i:i + batch_size]
                
                # Prepare batch for Pinecone (without record_hash)
                pinecone_batch = [(vid, emb, meta) for vid, emb, meta, _ in batch]
                index.upsert(pinecone_batch)
                
                # Save upload history
                for vid, _, meta, rec_hash in batch:
                    save_upload_history(rec_hash, meta['employee_original'], meta['date'], json_file, vid)
            
            print(f"✅ Uploaded {len(vectors)} new records from {json_file}")
            total_uploaded += len(vectors)
        else:
            print(f"ℹ️ No new records to upload from {json_file}")
        
        if skipped_count > 0:
            print(f"⏭️ Skipped {skipped_count} duplicate records")
            total_skipped += skipped_count
        
        # Save file status
        save_file_status(json_file, json_file, len(records), "completed")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 UPLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Total new records uploaded: {total_uploaded}")
    print(f"⏭️ Total duplicates skipped: {total_skipped}")
    print(f"📁 Files processed: {len(selected_files)}")
    print(f"{'='*60}\n")
    
    return True

# ----------------- Check Pinecone Stats ------------------
def check_pinecone_stats(index):
    """Check current stats of Pinecone index"""
    try:
        stats = index.describe_index_stats()
        print(f"\n📊 Pinecone Index Stats:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Index fullness: {stats.index_fullness}")
        if hasattr(stats, 'namespaces') and stats.namespaces:
            for namespace, info in stats.namespaces.items():
                print(f"   Namespace '{namespace}': {info.vector_count} vectors")
    except Exception as e:
        print(f"⚠️ Could not fetch Pinecone stats: {e}")

# ----------------- View Upload History ------------------
def view_upload_history():
    """View upload history from database"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""SELECT employee, date, source_file, upload_timestamp 
                 FROM upload_history 
                 ORDER BY upload_timestamp DESC 
                 LIMIT 20""")
    results = c.fetchall()
    conn.close()
    
    if results:
        print("\n📜 Recent Upload History (Last 20):")
        print("-" * 80)
        for emp, date, source, timestamp in results:
            print(f"   {emp:20} | {date:12} | {source:25} | {timestamp[:19]}")
    else:
        print("\n📜 No upload history found.")

# ----------------- Main Function ------------------
def main():
    """Main function with interactive menu"""
    print("\n" + "="*60)
    print("🚀 PINECONE TRAVEL RECORDS MANAGEMENT SYSTEM")
    print("="*60 + "\n")
    
    try:
        # Initialize database
        print("📊 Initializing database...")
        init_db()
        
        # Setup Pinecone
        print("🔗 Setting up Pinecone connection...")
        index, index_name = setup_pinecone()
        
        # Show menu
        while True:
            print("\n" + "-"*60)
            print("📋 MENU:")
            print("   1. Upload travel records (interactive)")
            print("   2. View Pinecone stats")
            print("   3. View upload history")
            print("   4. Exit")
            print("-"*60)
            
            choice = input("Select option (1-4): ").strip()
            
            if choice == "1":
                upload_travel_records_advanced(index)
            elif choice == "2":
                check_pinecone_stats(index)
            elif choice == "3":
                view_upload_history()
            elif choice == "4":
                print("\n👋 Exiting... Goodbye!")
                break
            else:
                print("❌ Invalid choice! Please select 1-4.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        raise

if __name__ == "__main__":
    main()