import os
import json
import sqlite3
import pickle
import re
import pandas as pd
from uuid import uuid4
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, flash
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import spacy
import time
import sqlite3
from datetime import datetime

# Load spaCy model (install: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env file.")
if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env file.")

# Flask app setup
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'sprl-temp-key-123'  # Simple temporary key
CORS(app)

# Gemini Setup
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL_NAME = "models/gemini-2.0-flash-exp"

# Pinecone Setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "travel-sprl"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global variables

MAX_STORED_MESSAGES = 50
last_person_name = None

def load_users_from_excel(excel_path="data/users.xlsx"):
    """Load users from Excel file"""
    try:
        if not os.path.exists(excel_path):
            print(f"⚠ Excel file not found: {excel_path}")
            return {}
        
        df = pd.read_excel(excel_path)
        users = {}
        
        # Assuming Excel has columns: email, password, name, department
        for _, row in df.iterrows():
            email = str(row.get('email', '')).lower().strip()
            password = str(row.get('password', '')).strip()
            name = str(row.get('name', '')).strip()
            department = str(row.get('department', '')).strip()
            
            if email and password:  # Only add if both email and password exist
                users[email] = {
                    'password': password,
                    'name': name,
                    'department': department
                }
        
        print(f"✅ Loaded {len(users)} users from Excel file")
        return users
        
    except Exception as e:
        print(f"❌ Error loading users from Excel: {e}")
        return {}


# ==================== DATABASE FUNCTIONS ====================


# Fixed init_db function

def init_db():
    """Initialize SQLite database with proper schema"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    # Check if users table exists and get its structure
    c.execute("PRAGMA table_info(users)")
    existing_columns = [column[1] for column in c.fetchall()]
    
    if not existing_columns:
        # Create users table if it doesn't exist
        print("🔧 Creating users table...")
        c.execute("""CREATE TABLE users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            last_login TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        print("✅ Users table created successfully")
    else:
        # Check if required columns exist, add them if missing
        print(f"🔍 Existing columns in users table: {existing_columns}")
        
        required_columns = {
            'password': 'TEXT NOT NULL',
            'name': 'TEXT NOT NULL', 
            'department': 'TEXT',
            'last_login': 'TEXT',
            'created_at': 'TEXT DEFAULT CURRENT_TIMESTAMP'
        }
        
        for column_name, column_def in required_columns.items():
            if column_name not in existing_columns:
                try:
                    print(f"➕ Adding missing column: {column_name}")
                    if 'NOT NULL' in column_def:
                        # For NOT NULL columns, add with a default value first
                        if column_name == 'password':
                            c.execute(f"ALTER TABLE users ADD COLUMN {column_name} TEXT DEFAULT 'temp123'")
                        elif column_name == 'name':
                            c.execute(f"ALTER TABLE users ADD COLUMN {column_name} TEXT DEFAULT 'Unknown User'")
                        else:
                            c.execute(f"ALTER TABLE users ADD COLUMN {column_name} TEXT")
                    else:
                        c.execute(f"ALTER TABLE users ADD COLUMN {column_name} {column_def}")
                    print(f"✅ Added column: {column_name}")
                except sqlite3.Error as e:
                    print(f"❌ Error adding column {column_name}: {e}")
    
    # User tokens table
    c.execute("""CREATE TABLE IF NOT EXISTS user_tokens (
        user_id TEXT PRIMARY KEY, 
        token_limit INT, 
        token_used INT, 
        token_remaining INT
    )""")
    
    # Chat history table
    c.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        user_id TEXT, 
        question TEXT, 
        answer TEXT, 
        timestamp TEXT,
        input_tokens INT,
        output_tokens INT
    )""")
    
    # Feedback table
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        user_id TEXT,
        original_query TEXT,
        original_response TEXT,
        corrected_answer TEXT
    )""")
    
    # User context table
    c.execute("""CREATE TABLE IF NOT EXISTS user_context (
        user_id TEXT PRIMARY KEY, 
        last_person TEXT
    )""")
    
    conn.commit()
    conn.close()
    print("✅ Database initialization completed")

def init_request_db():
    """Initialize request.db database with STANDARDIZED schema"""
    conn = sqlite3.connect("request.db")
    c = conn.cursor()
    
    # Token requests table - STANDARDIZED SCHEMA
    c.execute("""CREATE TABLE IF NOT EXISTS token_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT NOT NULL,
    user_email TEXT NOT NULL,
    user_department TEXT,
    tokens_requested INTEGER NOT NULL,
    current_tokens_used INTEGER DEFAULT 0,
    current_token_limit INTEGER DEFAULT 1000,
    token_remaining INTEGER DEFAULT 1000,
    new_token_limit INTEGER DEFAULT 1000,
    new_token_remaining INTEGER DEFAULT 1000,
    reason TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    request_timestamp TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    processed_by TEXT,
    processed_at TEXT
    )""")
    
    # Feedback table - STANDARDIZED SCHEMA
    c.execute("""CREATE TABLE IF NOT EXISTS incorrect_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        user_email TEXT NOT NULL,
        user_department TEXT,
        original_query TEXT NOT NULL,
        original_response TEXT NOT NULL,
        corrected_answer TEXT NOT NULL,
        feedback_timestamp TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'pending',
        reviewed_by TEXT,
        reviewed_at TEXT
    )""")
    
    # Admin actions log table
    c.execute("""CREATE TABLE IF NOT EXISTS admin_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        admin_email TEXT NOT NULL,
        action_type TEXT NOT NULL,
        target_user_email TEXT,
        details TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    
    conn.commit()
    conn.close()
    print("✅ Request database initialized with standardized schema")
    
def enhanced_pronoun_resolution(query, last_person, user_id):
    """Enhanced pronoun resolution with better pattern matching and context preservation"""
    if not last_person:
        print("⚠️ No person context available for pronoun resolution")
        return query, last_person
    
    print(f"🔍 Resolving pronouns for: '{query}' with context: {last_person}")
    
    normalized_query = query.strip()
    original_query = normalized_query
    
    # Enhanced pronoun replacement patterns with more specific matching
    pronoun_patterns = [
        # Email patterns - highest priority (FIXED: More comprehensive patterns)
        (r'\b(?:give|get|what(?:\s+is)?|tell\s+me)\s+(?:his|her)\s+email(?:\s+id)?\??', f"give {last_person}'s email"),
        (r'\bhis\s+email(?:\s+id)?\b', f"{last_person}'s email"),
        (r'\bher\s+email(?:\s+id)?\b', f"{last_person}'s email"),
        (r'\bemail\s+(?:id\s+)?(?:of\s+)?(?:him|her)\b', f"email of {last_person}"),
        (r'\bgive\s+(?:his|her)\s+email\b', f"give {last_person}'s email"),
        (r'\bwhat\s+(?:is\s+)?(?:his|her)\s+email\b', f"what is {last_person}'s email"),
        
        # Department patterns
        (r'\b(?:give|get|what(?:\s+is)?|tell\s+me)\s+(?:his|her)\s+department\b', f"give {last_person}'s department"),
        (r'\bhis\s+department\b', f"{last_person}'s department"),
        (r'\bher\s+department\b', f"{last_person}'s department"),
        (r'\bdepartment\s+(?:of\s+)?(?:him|her)\b', f"department of {last_person}"),
        
        # Contact information patterns
        (r'\bhis\s+(?:contact|details|information|info)\b', f"{last_person}'s contact information"),
        (r'\bher\s+(?:contact|details|information|info)\b', f"{last_person}'s contact information"),
        
        # Travel patterns
        (r'\bhis\s+(?:travel|trips|requests|journey|visit)\b', f"{last_person}'s travel"),
        (r'\bher\s+(?:travel|trips|requests|journey|visit)\b', f"{last_person}'s travel"),
        
        # General possessive patterns - be more specific to avoid over-replacement
        (r'\babout\s+him\b', f"about {last_person}"),
        (r'\babout\s+her\b', f"about {last_person}"),
        (r'\bfor\s+him\b', f"for {last_person}"),
        (r'\bfor\s+her\b', f"for {last_person}"),
        
        # Subject pronouns with actions
        (r'\bhe\s+(?:traveled|went|visited|requested|applied|submitted|works|is|was)\b', f"{last_person} "),
        (r'\bshe\s+(?:traveled|went|visited|requested|applied|submitted|works|is|was)\b', f"{last_person} "),
        
        # Generic pronouns (lower priority - only replace at word boundaries)
        (r'\bhis\b(?!\s+(?:email|department|travel|contact|details|information))', f"{last_person}'s"),
        (r'\bhers?\b(?!\s+(?:email|department|travel|contact|details|information))', f"{last_person}'s"),
        (r'\bhe\b(?!\s+(?:traveled|went|visited|requested|applied|submitted|works|is|was))', last_person),
        (r'\bshe\b(?!\s+(?:traveled|went|visited|requested|applied|submitted|works|is|was))', last_person),
    ]
    
    resolved_query = normalized_query
    replacements_made = []
    
    for pattern, replacement in pronoun_patterns:
        if re.search(pattern, resolved_query, re.IGNORECASE):
            old_query = resolved_query
            resolved_query = re.sub(pattern, replacement, resolved_query, flags=re.IGNORECASE)
            if old_query != resolved_query:
                replacements_made.append(f"'{pattern}' -> '{replacement}'")
                # Break after first successful replacement to avoid over-processing
                break
    
    if replacements_made:
        print(f"✅ Pronoun replacements made: {', '.join(replacements_made)}")
        print(f"   Original: '{original_query}'")
        print(f"   Resolved: '{resolved_query}'")
    else:
        print(f"ℹ️ No pronouns found to resolve in: '{query}'")
    
    # Return both resolved query and the person context to maintain it
    return resolved_query, last_person


def extract_person_names(text):
    """Extract person names from text using multiple strategies"""
    names = set()
    
    # Strategy 1: Simple regex for capitalized words (names)
    # Enhanced pattern to catch full names better
    simple_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    for name in simple_names:
        # Filter out common non-name words
        exclude_words = ['Travel', 'Request', 'Employee', 'Department', 'From', 'Date', 'Email', 'Cost', 
                        'Mumbai', 'Delhi', 'Chennai', 'Bangalore', 'India', 'Form', 'Manager', 'Director',
                        'Company', 'Office', 'Project', 'Team', 'Meeting', 'Report', 'Document']
        if name not in exclude_words and len(name) > 2:
            names.add(name.strip())
    
    # Strategy 2: Use spaCy if available for better name recognition
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.strip()) > 2:
                    names.add(ent.text.strip())
        except:
            pass
    
    # Strategy 3: Look for patterns like "Rohit Rajbhar", "John Smith" etc.
    full_name_pattern = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', text)
    for full_name in full_name_pattern:
        if len(full_name.strip()) > 3:
            names.add(full_name.strip())
    
    return list(names)

def smart_person_detection(query, chat_history=None):
    """Enhanced person detection with better context handling and clearer detection"""
    print(f"🔍 Detecting person in: '{query}'")
    
    # First check current query for names with enhanced patterns
    current_names = extract_person_names(query)
    
    # Enhanced name detection patterns
    name_patterns = [
        # Direct mention patterns
        r'\babout\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\bof\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\btell\s+me\s+about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\bwho\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:email|department|travel|info|information|details)\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:traveled|went|visited|works|is)\b',
    ]
    
    # Check for pattern-based name detection
    for pattern in name_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            # Filter out common non-name words
            exclude_words = ['Travel', 'Request', 'Employee', 'Department', 'From', 'Date', 'Email', 'Cost', 
                           'Mumbai', 'Delhi', 'Chennai', 'Bangalore', 'India', 'Form', 'Manager', 'Director',
                           'Company', 'Office', 'Project', 'Team', 'Meeting', 'Report', 'Document']
            if match not in exclude_words and len(match.strip()) > 2:
                current_names.append(match.strip())
    
    if current_names:
        # Return the most likely name (longest one or most complete)
        detected_person = max(current_names, key=lambda x: (len(x.split()), len(x)))
        print(f"🎯 Detected new person in current query: {detected_person}")
        return detected_person
    
    # If no names in current query, check recent chat history
    if chat_history:
        for prev_question, prev_answer in reversed(chat_history[-5:]):  # Check last 5 interactions
            prev_names = extract_person_names(prev_question + " " + prev_answer)
            if prev_names:
                detected_person = max(prev_names, key=lambda x: (len(x.split()), len(x)))
                print(f"🔍 Found person in chat history: {detected_person}")
                return detected_person
    
    return None


def has_pronouns(text):
    """Check if text contains pronouns that might need resolution"""
    # Enhanced pattern to catch more pronoun variations
    pronoun_patterns = [
        r'\b(?:he|she|him|her|his|hers|himself|herself)\b',
        r'\bemail\s+(?:id\s+)?(?:of\s+)?(?:him|her)\b',
        r'\bgive\s+(?:his|her)\s+email\b',
        r'\bwhat\s+(?:is\s+)?(?:his|her)\s+email\b',
        r'\b(?:about|for)\s+(?:him|her)\b'
    ]
    
    for pattern in pronoun_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def cleanup_old_corrections(days_old=30):
    """Clean up old corrections from Pinecone"""
    try:
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        index, _ = initialize_pinecone()
        
        if not index:
            return
        
        # Query old corrections
        results = index.query(
            vector=[0] * 384,  # Dummy vector
            top_k=1000,
            include_metadata=True,
            filter={"source": "user_correction"}
        )
        
        old_ids = []
        for match in results.get("matches", []):
            created_date = match.get("metadata", {}).get("updated_at", "")
            if created_date < cutoff_date:
                old_ids.append(match.get("id"))
        
        if old_ids:
            index.delete(ids=old_ids)
            print(f"🧹 Cleaned up {len(old_ids)} old corrections from Pinecone")
            
    except Exception as e:
        print(f"❌ Error cleaning up corrections: {e}")    

def log_admin_action(admin_email, action_type, target_user_email, details):
    """Log admin action"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        c.execute("""INSERT INTO admin_actions 
                     (admin_email, action_type, target_user_email, details, timestamp)
                     VALUES (?, ?, ?, ?, ?)""",
                  (admin_email, action_type, target_user_email, details, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error logging admin action: {e}")

def get_user_token_status(user_email):
    """Get current token usage and limit for a specific user from request.db"""
    try:
        conn = sqlite3.connect("request.db")
        cursor = conn.cursor()
        
        # Check if token_requests table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='token_requests'
        """)
        if not cursor.fetchone():
            conn.close()
            return {
                'current_usage': 0,
                'total_limit': 1000,
                'remaining': 1000,
                'status': 'active'
            }
        
        # Get the latest token status for this user
        cursor.execute('''
            SELECT 
                COALESCE(MAX(current_tokens_used), 0) as current_usage,
                COALESCE(MAX(current_token_limit), 1000) as total_limit
            FROM token_requests 
            WHERE LOWER(user_email) = LOWER(?)
            GROUP BY LOWER(user_email)
        ''', (user_email,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            current_usage = result[0]
            total_limit = result[1]
            remaining = max(0, total_limit - current_usage)
            
            # Determine status based on usage
            if remaining == 0:
                status = 'limit-reached'
            elif remaining <= total_limit * 0.1:  # Less than 10% remaining
                status = 'near-limit'
            else:
                status = 'active'
            
            return {
                'current_usage': current_usage,
                'total_limit': total_limit,
                'remaining': remaining,
                'status': status
            }
        else:
            # User not found in request database, return defaults
            return {
                'current_usage': 0,
                'total_limit': 1000,
                'remaining': 1000,
                'status': 'active'
            }
            
    except sqlite3.Error as e:
        print(f"Database error in get_user_token_status: {e}")
        return {
            'current_usage': 0,
            'total_limit': 1000,
            'remaining': 1000,
            'status': 'error'
        }
    except Exception as e:
        print(f"Error fetching user token status: {e}")
        return {
            'current_usage': 0,
            'total_limit': 1000,
            'remaining': 1000,
            'status': 'error'
        }

def update_user_token_usage(user_email, tokens_used):
    """Update user's token usage in the request database"""
    token_remaining_calc = 1000 - tokens_used
    new_token_limit = 1000
    new_token_remaining = token_remaining_calc
    try:
        conn = sqlite3.connect("request.db")
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='token_requests'
        """)
        if not cursor.fetchone():
            # Create table if it doesn't exist
            init_request_db()
        
        # Get current status
        cursor.execute('''
            SELECT current_tokens_used, current_token_limit 
            FROM token_requests 
            WHERE LOWER(user_email) = LOWER(?)
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (user_email,))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing record
            current_usage = result[0] + tokens_used
            current_limit = result[1]
            
            cursor.execute('''
                UPDATE token_requests 
                SET current_tokens_used = ?
                WHERE LOWER(user_email) = LOWER(?)
            ''', (current_usage, user_email))
        else:
            # Create new record if user doesn't exist
            cursor.execute('''
            INSERT INTO token_requests 
            (user_name, user_email, user_department, tokens_requested, 
            current_tokens_used, current_token_limit, token_remaining,
            new_token_limit, new_token_remaining, priority, reason, status, request_timestamp)
            VALUES (?, ?, ?, 0, ?, 1000, ?, 1000, ?, 'system', 'Token usage tracking', 'active', ?)
        ''', ('Unknown User', user_email, 'Unknown', tokens_used, token_remaining_calc, new_token_remaining, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error updating user token usage: {e}")
        return False
    
def migrate_token_requests_table():
    """Add new columns to existing token_requests table"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Check if new columns exist
        c.execute("PRAGMA table_info(token_requests)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'token_remaining' not in columns:
            c.execute("ALTER TABLE token_requests ADD COLUMN token_remaining INTEGER DEFAULT 0")
            c.execute("ALTER TABLE token_requests ADD COLUMN new_token_limit INTEGER DEFAULT 1000")
            c.execute("ALTER TABLE token_requests ADD COLUMN new_token_remaining INTEGER DEFAULT 1000")
            
            # Update existing records
            c.execute("""UPDATE token_requests SET 
                         token_remaining = current_token_limit - current_tokens_used,
                         new_token_limit = current_token_limit + tokens_requested,
                         new_token_remaining = (current_token_limit - current_tokens_used) + tokens_requested
                         WHERE token_remaining = 0""")
            
            conn.commit()
            print("✅ Migrated token_requests table with new columns")
        
        conn.close()
    except Exception as e:
        print(f"❌ Error migrating table: {e}")    

def update_pinecone_with_correction(original_query, original_response, corrected_answer):
    """Update Pinecone with corrected feedback"""
    try:
        index, _ = initialize_pinecone()
        if not index:
            print("❌ Cannot update Pinecone - no connection")
            return False
        
        # Create new corrected content
        corrected_text = (
            f"Query: {original_query}\n"
            f"Corrected Answer: {corrected_answer}\n"
            f"Source: User Correction\n"
            f"Updated: {datetime.now().isoformat()}"
        )
        
        # Generate embedding for the corrected content
        embedding = embedding_model.embed_query(corrected_text)
        
        # Create unique vector ID for correction
        correction_id = f"correction-{uuid4().hex[:8]}"
        
        # Metadata for the correction
        metadata = {
            "source": "user_correction",
            "type": "correction",
            "original_query": original_query,
            "corrected_answer": corrected_answer,
            "text": corrected_text,
            "updated_at": datetime.now().isoformat()
        }
        
        # Upsert to Pinecone
        index.upsert([(correction_id, embedding, metadata)])
        
        print(f"✅ Updated Pinecone with correction: {correction_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating Pinecone with correction: {e}")
        return False    

def save_token_request(user_name, user_email, user_department, tokens_requested, reason, priority):
    """Save token request to request.db - UPDATED for standardized schema"""
    token_limit, token_used, _ = get_user_tokens(user_email)
    token_remaining = token_limit - token_used
    
    # Get the latest new_token_limit from previous requests
    conn_check = sqlite3.connect("request.db")
    c_check = conn_check.cursor()
    c_check.execute("""SELECT new_token_limit FROM token_requests 
                       WHERE user_email = ? 
                       ORDER BY created_at DESC LIMIT 1""", (user_email,))
    last_request = c_check.fetchone()
    conn_check.close()
    current_token_limit = last_request[0] if last_request else token_limit
    
    # Use previous new_token_limit or current token_limit if no previous requests
    base_limit = last_request[0] if last_request else token_limit
    new_token_limit = base_limit + tokens_requested
    new_token_remaining = token_remaining + tokens_requested
    try:
        # Get current user token info
        token_limit, token_used, token_remaining = get_user_tokens(user_email)
        
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        request_timestamp = datetime.now().isoformat()
        
        # UPDATED: Using standardized column names
        c.execute("""INSERT INTO token_requests 
            (user_name, user_email, user_department, tokens_requested, 
            current_tokens_used, current_token_limit, token_remaining,
            new_token_limit, new_token_remaining, reason, priority, request_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_name, user_email, user_department, tokens_requested, 
        token_used, current_token_limit, token_remaining, new_token_limit,  # CHANGED
        new_token_remaining, reason, priority, request_timestamp))
        request_id = c.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Token request saved: ID {request_id} for {user_name} ({user_email})")
        return request_id
        
    except Exception as e:
        print(f"❌ Error saving token request: {e}")
        return None

def save_incorrect_feedback_request(user_name, user_email, user_department, original_query, original_response, corrected_answer):
    """Save incorrect feedback to request.db - UPDATED for standardized schema"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        feedback_timestamp = datetime.now().isoformat()
        
        # UPDATED: Using standardized column names
        c.execute("""INSERT INTO incorrect_feedback 
                     (user_name, user_email, user_department, original_query, 
                      original_response, corrected_answer, feedback_timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (user_name, user_email, user_department, original_query, 
                   original_response, corrected_answer, feedback_timestamp))
        
        feedback_id = c.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Incorrect feedback saved: ID {feedback_id} for {user_name} ({user_email})")
        return feedback_id
        
    except Exception as e:
        print(f"❌ Error saving incorrect feedback: {e}")
        return None

def get_pending_token_requests():
    """Get all pending token requests - UPDATED for standardized schema"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # UPDATED: Using standardized column names
        c.execute("""SELECT id, user_name, user_email, user_department, tokens_requested,
                    current_tokens_used, current_token_limit, token_remaining,
                    new_token_limit, new_token_remaining, reason, priority,
                    request_timestamp, status
             FROM token_requests 
             WHERE status = 'pending'
             ORDER BY request_timestamp DESC""")
        
        requests = []
        for row in c.fetchall():
            requests.append({
                "id": row[0],
                "user_name": row[1],
                "user_email": row[2],
                "user_department": row[3],
                "tokens_requested": row[4],
                "current_tokens_used": row[5],
                "current_token_limit": row[6],
                "token_remaining": row[7],
                "new_token_limit": row[8],
                "new_token_remaining": row[9],
                "reason": row[10],
                "priority": row[11],
                "request_timestamp": row[12],
                "status": row[13],
                "current_usage": f"{row[5]} / {row[6]} tokens"
            })
        
        conn.close()
        return requests
        
    except Exception as e:
        print(f"❌ Error getting token requests: {e}")
        return []

def get_pending_feedback_requests():
    """Get all pending feedback requests - UPDATED for standardized schema"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # UPDATED: Using standardized column names
        c.execute("""SELECT id, user_name, user_email, user_department, original_query,
                            original_response, corrected_answer, feedback_timestamp, status
                     FROM incorrect_feedback 
                     WHERE status = 'pending'
                     ORDER BY feedback_timestamp DESC""")
        
        feedback_list = []
        for row in c.fetchall():
            feedback_list.append({
                "id": row[0],
                "user_name": row[1],           
                "user_email": row[2],          
                "user_department": row[3],     
                "original_query": row[4],      
                "original_response": row[5],   
                "corrected_answer": row[6],    
                "feedback_timestamp": row[7],  
                "status": row[8]
            })
        
        conn.close()
        return feedback_list
        
    except Exception as e:
        print(f"❌ Error getting feedback requests: {e}")
        return []

def update_token_request_status(request_id, status, admin_email, tokens_granted=0):
    """Enhanced token request update with immediate token grant"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        c.execute("SELECT user_email, tokens_requested FROM token_requests WHERE id = ?", (request_id,))
        request_data = c.fetchone()
        
        if not request_data:
            conn.close()
            return False, "Request not found"
        
        user_email, tokens_requested = request_data
        processed_at = datetime.now().isoformat()
        
        # Update request status
        c.execute("""UPDATE token_requests 
                     SET status = ?, processed_by = ?, processed_at = ?
                     WHERE id = ?""",
                  (status, admin_email, processed_at, request_id))
        
        # ✅ NEW: If approved, immediately grant tokens
        if status == 'approved' and tokens_granted > 0:
            print(f"🔄 Auto-granting {tokens_granted} tokens to {user_email}...")
            
            # Update user tokens in main database immediately
            token_limit, token_used, token_remaining = get_user_tokens(user_email)
            new_limit = token_limit + tokens_granted
            new_remaining = token_remaining + tokens_granted  # Add tokens to remaining, not just limit
            update_user_token_limit(user_email, new_limit)

            c.execute("""UPDATE token_requests
             SET current_tokens_used = ?, current_token_limit = ?
             WHERE id = ?""",
          (token_used, new_limit, request_id))
            
            # Log admin action
            log_admin_action(admin_email, "token_grant", user_email, 
                           f"Auto-granted {tokens_granted} tokens (Request ID: {request_id})")
            
            print(f"✅ Tokens granted immediately: {tokens_granted} to {user_email}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Token request {request_id} updated to {status} by {admin_email}")
        return True, "Request updated successfully"
        
    except Exception as e:
        print(f"❌ Error updating token request: {e}")
        return False, str(e)
    
def update_feedback_status(feedback_id, status, admin_email):
    """Enhanced feedback status update with automatic Pinecone sync"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get feedback details before updating
        c.execute("""SELECT original_query, original_response, corrected_answer, user_email
                     FROM incorrect_feedback WHERE id = ?""", (feedback_id,))
        feedback_data = c.fetchone()
        
        if not feedback_data:
            conn.close()
            return False, "Feedback not found"
        
        original_query, original_response, corrected_answer, user_email = feedback_data
        reviewed_at = datetime.now().isoformat()
        
        # Update feedback status
        c.execute("""UPDATE incorrect_feedback 
                     SET status = ?, reviewed_by = ?, reviewed_at = ?
                     WHERE id = ?""",
                  (status, admin_email, reviewed_at, feedback_id))
        
        # ✅ NEW: If approved, immediately sync to Pinecone
        if status in ['approved', 'implemented']:
            print(f"🔄 Auto-syncing approved feedback to Pinecone...")
            
            # Update Pinecone with the correction
            pinecone_success = update_pinecone_with_correction(original_query, original_response, corrected_answer)
            c.execute("SELECT token_used, token_remaining, token_limit FROM user_tokens WHERE user_id = ?", (user_id,))
            current_data = c.fetchone()
            used, old_remaining, old_limit = current_data if current_data else (0, 0, 0)
            new_limit=old_limit+tokens_added
            # Calculate new remaining tokens (add the difference to existing remaining)
            tokens_added = new_limit - old_limit
            new_remaining = old_remaining + tokens_added
            if pinecone_success:
                # Also update the user's local feedback in users.db for immediate use
                conn_users = sqlite3.connect("users.db")
                c_users = conn_users.cursor()
                c_users.execute("UPDATE user_tokens SET token_remaining = ? WHERE user_id = ?", 
                                (new_remaining, user_email))
                conn_users.commit()
                conn_users.close()
                
                details = f"Approved feedback ID: {feedback_id} and auto-synced to Pinecone"
                print(f"✅ Feedback auto-synced to Pinecone successfully")
            else:
                details = f"Approved feedback ID: {feedback_id} but Pinecone sync failed"
                print(f"❌ Pinecone sync failed for feedback ID: {feedback_id}")
        else:
            details = f"Reviewed feedback ID: {feedback_id}, Status: {status}"
        
        # Log admin action
        log_admin_action(admin_email, "feedback_review", user_email, details)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Feedback {feedback_id} updated to {status} by {admin_email}")
        return True, "Feedback updated successfully"
        
    except Exception as e:
        print(f"❌ Error updating feedback: {e}")
        return False, str(e)
    
def get_user_tokens(user_id):
    """Get user token information from request.db"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get the latest token status from request.db
        c.execute('''
            SELECT new_token_limit, current_tokens_used, new_token_remaining 
            FROM token_requests 
            WHERE LOWER(user_email) = LOWER(?)
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (user_id,))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            token_limit, token_used, token_remaining = result
            return (token_limit, token_used, token_remaining)
        else:
            # If no record found, return defaults and create one
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO user_tokens VALUES (?, ?, ?, ?)", (user_id, 1000, 0, 1000))
            conn.commit()
            conn.close()
            return (1000, 0, 1000)
            
    except Exception as e:
        print(f"Error getting user tokens: {e}")
        return (1000, 0, 1000)

def update_user_tokens(user_id, tokens_used):
    """Update user token usage in request.db"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get current status
        c.execute('''
            SELECT current_tokens_used, new_token_limit, new_token_remaining
            FROM token_requests 
            WHERE LOWER(user_email) = LOWER(?)
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (user_id,))
        
        result = c.fetchone()
        
        if result:
            current_used, token_limit, current_remaining = result
            new_used = current_used + tokens_used
            new_remaining = max(0, current_remaining - tokens_used)
            
            # Update the latest record
            c.execute('''
                UPDATE token_requests 
                SET current_tokens_used = ?, new_token_remaining = ?
                WHERE user_email = ? AND id = (
                    SELECT id FROM token_requests 
                    WHERE user_email = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                )
            ''', (new_used, new_remaining, user_id, user_id))
            
        else:
            # Create new record if none exists
            new_used = tokens_used
            new_remaining = max(0, 1000 - tokens_used)
            
            c.execute('''
                INSERT INTO token_requests 
                (user_name, user_email, user_department, tokens_requested, 
                current_tokens_used, current_token_limit, token_remaining,
                new_token_limit, new_token_remaining, priority, reason, status, request_timestamp)
                VALUES (?, ?, ?, 0, ?, 1000, ?, 1000, ?, 'system', 'Token usage tracking', 'active', ?)
            ''', ('Unknown User', user_id, 'Unknown', new_used, 1000-new_used, new_remaining, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"Token usage updated: -{tokens_used}, Remaining: {new_remaining}")
        
    except Exception as e:
        print(f"Error updating user tokens: {e}")

def save_chat_history(user_id, question, answer, input_tokens=0, output_tokens=0):
    """Save chat history to database"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO chat_history VALUES (?, ?, ?, ?, ?, ?)", 
              (user_id, question, answer, timestamp, input_tokens, output_tokens))
    conn.commit()
    conn.close()

def get_chat_history(user_id, limit=10):
    """Get recent chat history for user"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""SELECT question, answer FROM chat_history 
                 WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?""", (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return [(q, a) for q, a in reversed(rows)]

def get_last_person(user_id):
    """Get last person mentioned by user"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT last_person FROM user_context WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    
    if not result:
        c.execute("INSERT INTO user_context VALUES (?, ?)", (user_id, None))
        conn.commit()
        conn.close()
        return None
    
    conn.close()
    return result[0]

def set_last_person(user_id, person_name):
    """Set last person mentioned by user"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO user_context VALUES (?, ?)", (user_id, person_name))
    conn.commit()
    conn.close()

def save_feedback(user_id, original_query, original_response, corrected_answer):
    """Save user feedback"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?)", 
              (user_id, original_query, original_response, corrected_answer))
    conn.commit()
    conn.close()

def get_corrected_feedback(user_id, query):
    """Get corrected feedback for a query with fuzzy matching"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    # First try exact match
    c.execute("SELECT corrected_answer FROM feedback WHERE user_id = ? AND original_query = ?", 
              (user_id, query))
    result = c.fetchone()
    
    if result:
        conn.close()
        return result[0]
    
    # If no exact match, try to find similar queries (optional advanced feature)
    # This could use fuzzy string matching if you want to implement it
    # For now, just return None if no exact match
    
    conn.close()
    return None

# ==================== UTILITY FUNCTIONS ====================

def count_tokens_approx(text: str) -> int:
    """Approximate token counting"""
    return len(text) // 4

# Add these functions to your Flask app to sync users.xlsx with users.db

def sync_users_excel_to_db():
    """Fixed sync users from Excel file to database"""
    try:
        # Load users from Excel
        excel_users = load_users_from_excel("data/users.xlsx")
        if not excel_users:
            print("❌ No users found in Excel file")
            return False
        
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        # Get current users from database
        c.execute("SELECT email FROM users")
        db_users = set(row[0] for row in c.fetchall())
        excel_user_emails = set(excel_users.keys())
        
        # Track changes
        added_users = 0
        updated_users = 0
        deleted_users = 0
        
        # Add or update users from Excel
        for email, user_info in excel_users.items():
            # Check if user exists in database
            c.execute("SELECT email, name, department, password FROM users WHERE email = ?", (email,))
            existing_user = c.fetchone()
            
            current_time = datetime.now().isoformat()
            
            if existing_user:
                # Update existing user if data has changed
                existing_name = existing_user[1] if len(existing_user) > 1 else ''
                existing_dept = existing_user[2] if len(existing_user) > 2 else ''
                existing_pass = existing_user[3] if len(existing_user) > 3 else ''
                
                if (existing_name != user_info['name'] or 
                    existing_dept != user_info.get('department', '') or
                    existing_pass != user_info['password']):
                    
                    try:
                        c.execute("""UPDATE users 
                                    SET name = ?, department = ?, password = ?
                                    WHERE email = ?""",
                                 (user_info['name'], user_info.get('department', ''), 
                                  user_info['password'], email))
                        updated_users += 1
                        print(f"🔄 Updated user: {user_info['name']} ({email})")
                    except sqlite3.Error as e:
                        print(f"❌ Error updating user {email}: {e}")
            else:
                # Add new user
                try:
                    c.execute("""INSERT INTO users 
                                (email, password, name, department, created_at)
                                VALUES (?, ?, ?, ?, ?)""",
                             (email, user_info['password'], user_info['name'], 
                              user_info.get('department', ''), current_time))
                    
                    # Initialize user tokens for new user
                    c.execute("INSERT OR IGNORE INTO user_tokens VALUES (?, ?, ?, ?)", 
                             (email, 1000, 0, 1000))
                    
                    added_users += 1
                    print(f"➕ Added new user: {user_info['name']} ({email})")
                except sqlite3.Error as e:
                    print(f"❌ Error adding user {email}: {e}")
        
        # Remove users that are no longer in Excel
        users_to_delete = db_users - excel_user_emails
        for email_to_delete in users_to_delete:
            try:
                c.execute("DELETE FROM users WHERE email = ?", (email_to_delete,))
                # Also clean up their data
                c.execute("DELETE FROM user_tokens WHERE user_id = ?", (email_to_delete,))
                c.execute("DELETE FROM chat_history WHERE user_id = ?", (email_to_delete,))
                c.execute("DELETE FROM user_context WHERE user_id = ?", (email_to_delete,))
                deleted_users += 1
                print(f"🗑️ Deleted user: {email_to_delete}")
            except sqlite3.Error as e:
                print(f"❌ Error deleting user {email_to_delete}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ User sync completed:")
        print(f"   📊 Total users in Excel: {len(excel_users)}")
        print(f"   ➕ Added: {added_users}")
        print(f"   🔄 Updated: {updated_users}")
        print(f"   🗑️ Deleted: {deleted_users}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error syncing users: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def fix_existing_users_table():
    """Fix existing users table structure if it's incompatible"""
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        # Check current table structure
        c.execute("PRAGMA table_info(users)")
        columns = c.fetchall()
        print(f"🔍 Current users table structure: {columns}")
        
        # Get all existing data
        c.execute("SELECT * FROM users")
        existing_data = c.fetchall()
        print(f"📊 Found {len(existing_data)} existing users")
        
        if columns and len(columns) > 0:
            # Check if password column exists
            column_names = [col[1] for col in columns]
            if 'password' not in column_names:
                print("🔧 Password column missing, recreating table...")
                
                # Backup existing data
                backup_data = []
                for row in existing_data:
                    # Try to extract email (should be first column)
                    email = row[0] if len(row) > 0 else f"user_{len(backup_data)}@company.com"
                    name = row[1] if len(row) > 1 else "Unknown User"
                    backup_data.append((email, name))
                
                # Drop and recreate table
                c.execute("DROP TABLE users")
                c.execute("""CREATE TABLE users (
                    email TEXT PRIMARY KEY,
                    password TEXT NOT NULL DEFAULT 'temp123',
                    name TEXT NOT NULL DEFAULT 'Unknown User',
                    department TEXT DEFAULT '',
                    last_login TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )""")
                
                # Restore data with default password
                for email, name in backup_data:
                    c.execute("""INSERT INTO users (email, password, name, department, created_at)
                                VALUES (?, ?, ?, ?, ?)""",
                             (email, 'temp123', name, '', datetime.now().isoformat()))
                
                print(f"✅ Recreated users table and restored {len(backup_data)} users with default password")
                print("⚠️  All users have default password 'temp123' - they will be updated from Excel on next sync")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing users table: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def reset_database():
    """Reset the entire database (use with caution)"""
    try:
        import os
        if os.path.exists("users.db"):
            os.remove("users.db")
            print("🗑️ Removed old users.db")
        
        # Recreate fresh database
        init_db()
        print("✅ Created fresh database")
        return True
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        return False

def check_user_sync_needed():
    """Check if Excel file is newer than last sync"""
    try:
        excel_path = "data/users.xlsx"
        sync_flag_path = "users_sync.flag"
        
        if not os.path.exists(excel_path):
            print("⚠️ users.xlsx not found")
            return False
        
        excel_mtime = os.path.getmtime(excel_path)
        
        if os.path.exists(sync_flag_path):
            with open(sync_flag_path, 'r') as f:
                last_sync_time = float(f.read().strip())
            
            # If Excel file is newer than last sync
            if excel_mtime > last_sync_time:
                print("🔄 Excel file has been modified, syncing users...")
                return True
            else:
                print("✅ Users already synced")
                return False
        else:
            print("🆕 First time sync needed")
            return True
            
    except Exception as e:
        print(f"❌ Error checking sync status: {e}")
        return True  # Force sync on error

def update_sync_flag():
    """Update the sync timestamp flag"""
    try:
        sync_flag_path = "users_sync.flag"
        current_time = time.time()
        with open(sync_flag_path, 'w') as f:
            f.write(str(current_time))
    except Exception as e:
        print(f"❌ Error updating sync flag: {e}")
        
def initialize_system():
    """Enhanced system initialization with error handling"""
    try:
        print("🚀 Initializing SPRL Chatbot with Pinecone + Gemini...")
        
        # Step 1: Check if database needs fixing
        try:
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            table_exists = c.fetchone() is not None
            
            if table_exists:
                c.execute("PRAGMA table_info(users)")
                columns = [col[1] for col in c.fetchall()]
                if 'password' not in columns:
                    print("🔧 Users table needs fixing...")
                    conn.close()
                    fix_existing_users_table()
                else:
                    conn.close()
            else:
                conn.close()
                print("🆕 No users table found, will create new one")
                
        except Exception as e:
            print(f"⚠️ Database check failed: {e}, creating fresh database...")
            if conn:
                conn.close()
        
        # Step 2: Initialize databases
        print("🗄️ Initializing databases...")
        init_db()
        init_request_db()
        migrate_token_requests_table()
        
        # Step 3: Sync users from Excel to database
        print("👥 Syncing users from Excel to database...")
        if check_user_sync_needed():
            sync_success = sync_users_excel_to_db()
            if sync_success:
                update_sync_flag()
                print("✅ Users synced successfully")
            else:
                print("⚠️ User sync failed - checking if Excel file exists...")
                if not os.path.exists("data/users.xlsx"):
                    print("❌ users.xlsx not found in data/ directory")
                    print("   Please create data/users.xlsx with columns: email, password, name, department")
                    return False
        else:
            print("✅ Users already synced")
        
        # Step 4: Verify user sync worked
        try:
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM users")
            user_count = c.fetchone()[0]
            conn.close()
            print(f"📊 Database now has {user_count} users")
        except Exception as e:
            print(f"⚠️ Could not verify user count: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_all_active_users():
    """Get all active users from database (synced from Excel)"""
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        c.execute("""SELECT email, name, department, created_at, last_login
                     FROM users ORDER BY name""")
        
        users = []
        for row in c.fetchall():
            # Get token info for each user
            c.execute("SELECT token_limit, token_used, token_remaining FROM user_tokens WHERE user_id = ?", (row[0],))
            token_data = c.fetchone()
            
            users.append({
                "email": row[0],
                "name": row[1],
                "department": row[2],
                "created_at": row[3],
                "last_login": row[4],
                "token_limit": token_data[0] if token_data else 1000,
                "token_used": token_data[1] if token_data else 0,
                "token_remaining": token_data[2] if token_data else 1000
            })
        
        conn.close()
        return users
        
    except Exception as e:
        print(f"❌ Error getting active users: {e}")
        return []

def update_user_last_login(email):
    """Update last login timestamp for user"""
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        current_time = datetime.now().isoformat()
        c.execute("UPDATE users SET last_login = ? WHERE email = ?", (current_time, email))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error updating last login: {e}")

def resolve_coreferences(context, question):
    """From backend - line 18"""
    if not nlp:
        return question
    try:
        doc = nlp(context + " " + question)
        return doc.coref_resolved if hasattr(doc, 'coref_resolved') else question
    except:
        return question

def search_employee_records(employee_name: str, query_text: str = "") -> list:
    """Enhanced search function from backend - line 122"""
    print(f"🔍 Searching for employee: '{employee_name}'")
    
    name_lower = employee_name.lower().strip()
    name_parts = name_lower.split()
    
    all_results = []
    search_strategies = []
    
    # Strategy 1: Exact name match (lowercase)
    search_strategies.append(("exact_lower", name_lower))
    
    # Strategy 2: Title case
    search_strategies.append(("title_case", employee_name.title()))
    
    # Strategy 3: Individual name parts
    for part in name_parts:
        if len(part) > 2:
            search_strategies.append(("name_part", part))
    
    # Strategy 4: First + Last name only (if more than 2 parts)
    if len(name_parts) >= 2:
        first_last = f"{name_parts[0]} {name_parts[-1]}"
        search_strategies.append(("first_last", first_last))
    
    for strategy_name, search_term in search_strategies:
        try:
            print(f"   Trying {strategy_name}: '{search_term}'")
            
            # Try with employee filter
            results_filtered = index.query(
                vector=embedding_model.embed_query(f"employee {search_term} travel request"),
                top_k=20,
                include_metadata=True,
                filter={"employee": search_term}
            )
            
            matches = results_filtered.get("matches", [])
            print(f"   Found {len(matches)} matches with filter")
            all_results.extend(matches)
            index, _ = initialize_pinecone()
            if not index:
                return
            # Try without filter but check content
            results_broad = index.query(
                vector=embedding_model.embed_query(f"{search_term} employee travel"),
                top_k=30,
                include_metadata=True
            )
            
            # Filter results that contain the name
            for match in results_broad.get("matches", []):
                content = match.get("metadata", {}).get("text", "").lower()
                if search_term.lower() in content:
                    all_results.append(match)
            
        except Exception as e:
            print(f"   Error in {strategy_name}: {e}")
            continue
    
    # Remove duplicates based on vector ID
    seen_ids = set()
    unique_results = []
    for result in all_results:
        result_id = result.get("id", "")
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            unique_results.append(result)
    
    return unique_results

def count_employee_travels(employee_name: str) -> int:
    """Count total travel records for an employee - from backend line 179"""
    results = search_employee_records(employee_name)
    return len(results)

def resolve_coreferences(context, question):
    """Resolve coreferences in question (requires spaCy model with coref)"""
    if not nlp:
        return question
    try:
        doc = nlp(context + " " + question)
        return doc.coref_resolved if hasattr(doc, 'coref_resolved') else question
    except:
        return question

def split_questions(text):
    """Split multiple questions in one string"""
    questions = re.findall(r'[^?]*\?', text)
    return [q.strip() for q in questions if q.strip()]



# ==================== PINECONE FUNCTIONS ====================

def initialize_pinecone():
    """Match backend initialization"""
    try:
        if index_name not in pc.list_indexes().names():
            print(f"❌ Pinecone index '{index_name}' not found. Creating new index...")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
            )
            print(f"✅ Created new Pinecone index: {index_name}")
        
        index = pc.Index(index_name)
        docsearch = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            text_key="text"
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 50})
        
        return index, retriever
    except Exception as e:
        print(f"❌ Pinecone initialization failed: {e}")
        return None, None
    
def gemini_generate_response(query: str, context_docs: list) -> str:
    """Updated to match backend system instructions"""
    system_instructions = (
        "You are an assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise.\n\n"
    )
    
    # Handle different context formats
    context_texts = []
    for doc in context_docs:
        if hasattr(doc, 'page_content'):
            context_texts.append(doc.page_content)
        elif isinstance(doc, str):
            context_texts.append(doc)
        else:
            context_texts.append(str(doc))
    
    combined_context = "\n\n".join(context_texts)
    prompt = f"{system_instructions}{combined_context}\n\nQuestion: {query}"

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,  # Match backend temperature
                "max_output_tokens": 1024
            }
        )
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return "Error: Gemini failed."
# ==================== MAIN LOGIC ====================

def process_question(query, user_id):
    """Enhanced process_question with fixed pronoun resolution and context management"""
    input_tokens = count_tokens_approx(query)
    token_limit, token_used, remaining = get_user_tokens(user_id)
    
    if remaining < input_tokens + 100:
        return "❌ Not enough tokens remaining. Please contact administrator."
    
    # Check for corrected feedback first
    corrected = get_corrected_feedback(user_id, query)
    if corrected:
        output_tokens = count_tokens_approx(corrected)
        total_tokens = input_tokens + output_tokens
        save_chat_history(user_id, query, corrected, input_tokens, output_tokens)
        update_user_tokens(user_id, total_tokens)
        return corrected

    # Initialize Pinecone
    index, retriever = initialize_pinecone()
    if not index or not retriever:
        return "❌ Search system unavailable. Please try again later."

    # Get chat history for better context
    chat_history = get_chat_history(user_id, limit=10)
    
    # Get last person context
    last_person = get_last_person(user_id)
    print(f"📋 Current person context: {last_person}")
    
    # Store original query for saving to history
    original_query = query
    
    # Step 1: Enhanced person detection - check for NEW person mentions first
    detected_person = smart_person_detection(query, chat_history)
    if detected_person:
        # IMPORTANT: Only update context if this is a NEW person mention, not a pronoun
        if not has_pronouns(query):  # Only update if query doesn't contain pronouns
            set_last_person(user_id, detected_person)
            last_person = detected_person
            print(f"🎯 Updated context with new person: {detected_person}")
        else:
            print(f"🔍 Found person '{detected_person}' but query has pronouns, keeping existing context: {last_person}")
    
    # Step 2: Enhanced pronoun resolution - only if we have a person context and query has pronouns
    if last_person and has_pronouns(query):
        print(f"🔄 Applying pronoun resolution with context: {last_person}")
        query, maintained_person = enhanced_pronoun_resolution(query, last_person, user_id)
        print(f"🔄 Query after pronoun resolution: '{query}'")
        
        # Ensure the person context is maintained after resolution
        if maintained_person:
            set_last_person(user_id, maintained_person)
    else:
        if not last_person:
            print("ℹ️ No person context available for pronoun resolution")
        elif not has_pronouns(query):
            print("ℹ️ No pronouns detected in query")

    # Handle "how many times" queries with better pattern matching
    count_patterns = [
        r"how many times (?:did|has) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) (?:travel|go|visit)",
        r"how many (?:trips|travels|journeys) (?:did|has) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"count.*(?:trips|travels).*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*).*how many times"
    ]
    
    for pattern in count_patterns:
        count_match = re.search(pattern, query, re.IGNORECASE)
        if count_match:
            person = count_match.group(1).strip()
            set_last_person(user_id, person)

            num_trips = count_employee_travels(person)
            answer = f"{person} traveled {num_trips} times."

            output_tokens = count_tokens_approx(answer)
            total_tokens = input_tokens + output_tokens
            save_chat_history(user_id, original_query, answer, input_tokens, output_tokens)
            update_user_tokens(user_id, total_tokens)
            return answer

    # Split multiple questions
    questions = split_questions(query) if '?' in query else [query]
    answers = []

    for q in questions:
        context_texts = []
        
        # Enhanced person search - try multiple strategies
        person_for_search = None
        
        # 1. Check if query mentions a specific person
        person_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", q)
        if person_match:
            potential_person = person_match.group(1).strip()
            # Verify it's actually a person name, not a place or common word
            exclude_words = ['Travel', 'Request', 'Email', 'Department', 'Mumbai', 'Delhi', 'Chennai', 'Bangalore']
            if potential_person not in exclude_words:
                person_for_search = potential_person
                # Update context with this person
                set_last_person(user_id, person_for_search)
                print(f"🎯 Found person in query, updating context: {person_for_search}")
        
        # 2. Use last known person if query has pronouns or no specific person mentioned
        elif last_person:
            person_for_search = last_person
            print(f"🔄 Using last known person for search: {person_for_search}")
            
        if person_for_search:
            print(f"🔍 Searching for employee: {person_for_search}")
            
            # Use enhanced search with the resolved person name
            search_results = search_employee_records(person_for_search, q)
            for result in search_results:
                content = result.get("metadata", {}).get("text", "")
                if content:
                    context_texts.append(content)
            
            print(f"📊 Found {len(context_texts)} context documents for {person_for_search}")
        
        # If no specific person or no results, use general retrieval
        if not context_texts:
            print(f"🔍 Using general search for query: {q}")
            docs = retriever.invoke(q)
            context_texts = [doc.page_content for doc in docs]

        # Generate answer
        answer = gemini_generate_response(q, context_texts)
        
        if len(questions) > 1:
            answers.append(f"Q: {q}\nA: {answer}")
        else:
            answers.append(answer)

    final_answer = "\n\n".join(answers)
    output_tokens = count_tokens_approx(final_answer)
    total_tokens = input_tokens + output_tokens

    save_chat_history(user_id, original_query, final_answer, input_tokens, output_tokens)
    update_user_tokens(user_id, total_tokens)

    # Maintain person context after processing
    if last_person:
        print(f"💾 Maintaining person context: {last_person}")

    return final_answer

def debug_context(user_id):
    """Debug function to check current context"""
    last_person = get_last_person(user_id)
    chat_history = get_chat_history(user_id, limit=5)
    
    print(f"🔍 Debug Context for {user_id}:")
    print(f"   Last Person: {last_person}")
    print(f"   Recent Chat History ({len(chat_history)} items):")
    for i, (q, a) in enumerate(chat_history):
        print(f"     {i+1}. Q: {q[:50]}...")
        print(f"        A: {a[:50]}...")

# ==================== ADDITIONAL DATABASE FUNCTIONS ====================

def get_all_users_token_info():
    """Get token information for all users"""
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    # Get all users from Excel and their token info
    users = load_users_from_excel("data/users.xlsx")
    user_tokens = []
    
    for email, user_info in users.items():
        c.execute("SELECT token_limit, token_used, token_remaining FROM user_tokens WHERE user_id = ?", (email,))
        token_data = c.fetchone()
        
        if not token_data:
            # Create new user with default tokens if not exists
            c.execute("INSERT INTO user_tokens VALUES (?, ?, ?, ?)", (email, 10000, 0, 10000))
            conn.commit()
            token_data = (10000, 0, 10000)
        
        user_tokens.append({
            "email": email,
            "name": user_info['name'],
            "department": user_info['department'],
            "token_limit": token_data[0],
            "token_used": token_data[1],
            "token_remaining": token_data[2],
            "usage_percentage": (token_data[1] / token_data[0] * 100) if token_data[0] > 0 else 0
        })
    
    conn.close()
    return user_tokens

def update_user_token_limit(user_id, new_limit):
    """Update user token limit in request.db"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get current usage
        c.execute('''
            SELECT current_tokens_used, new_token_remaining, new_token_limit
            FROM token_requests 
            WHERE LOWER(user_email) = LOWER(?)
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (user_id,))
        
        result = c.fetchone()
        
        if result:
            current_used, old_remaining, old_limit = result
            tokens_added = new_limit - old_limit
            new_remaining = old_remaining + tokens_added
            
            # Update the latest record
            c.execute('''
                UPDATE token_requests 
                SET new_token_limit = ?, new_token_remaining = ?
                WHERE user_email = ? AND id = (
                    SELECT id FROM token_requests 
                    WHERE user_email = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                )
            ''', (new_limit, new_remaining, user_id, user_id))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error updating token limit: {e}")

def get_token_requests():
    """Get all pending token requests from requests.db"""
    import sqlite3
    
    try:
        # Connect to requests.db database
        conn = sqlite3.connect('requests.db')
        cursor = conn.cursor()
        
        # Query to get all pending token requests
        cursor.execute("""
            SELECT id, employee, email, department, tokens_requested, 
                   current_usage, priority, duration, reason, timestamp
            FROM token_requests 
            WHERE status = 'pending'
            ORDER BY timestamp DESC
        """)
        
        # Fetch all results
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        requests = []
        for row in rows:
            requests.append({
                "id": row[0],
                "employee": row[1],
                "email": row[2],
                "department": row[3],
                "tokens_requested": row[4],
                "current_usage": row[5],
                "priority": row[6],
                "duration": row[7],
                "reason": row[8],
                "timestamp": row[9]
            })
        
        conn.close()
        return requests
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

# ==================== FIXED FLASK ROUTES FOR PROPER LOGIN FLOW ====================
@app.route('/api/user-token-status', methods=['GET'])
def api_user_token_status():
    """Get current user's token status"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    user_email = session.get('user_email')
    if not user_email:
        return jsonify({'success': False, 'error': 'User email not found in session'}), 400
    
    token_status = get_user_token_status(user_email)
    
    return jsonify({
        'success': True,
        'token_status': token_status
    })

@app.route('/api/update-token-usage', methods=['POST'])
def api_update_token_usage():
    """Update user's token usage"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        tokens_used = data.get('tokens_used', 0)
        user_email = session.get('user_email')
        
        if not user_email:
            return jsonify({'success': False, 'error': 'User email not found in session'}), 400
        
        if not isinstance(tokens_used, (int, float)) or tokens_used < 0:
            return jsonify({'success': False, 'error': 'Invalid token usage amount'}), 400
        
        success = update_user_token_usage(user_email, int(tokens_used))
        
        if success:
            # Return updated status
            token_status = get_user_token_status(user_email)
            return jsonify({
                'success': True,
                'message': f'Token usage updated: {tokens_used} tokens used',
                'token_status': token_status
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to update token usage'}), 500
            
    except Exception as e:
        print(f"Error in update token usage API: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/api/refresh-tokens', methods=['POST'])
def api_refresh_tokens():
    """Refresh user's token quota (for recharge functionality)"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        user_email = session.get('user_email')
        if not user_email:
            return jsonify({'success': False, 'error': 'User email not found in session'}), 400
        
        # Get current token status
        token_status = get_user_token_status(user_email)
        
        # Check if user has any pending token requests
        conn = sqlite3.connect("request.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM token_requests 
            WHERE LOWER(user_email) = LOWER(?) AND status = 'pending'
        ''', (user_email,))
        
        pending_requests = cursor.fetchone()[0]
        conn.close()
        
        if pending_requests > 0:
            return jsonify({
                'success': False,
                'error': f'You have {pending_requests} pending token request(s). Please wait for admin approval.',
                'token_status': token_status
            }), 400
        
        # If tokens are not fully exhausted, just refresh the display
        if token_status['remaining'] > 0:
            return jsonify({
                'success': True,
                'message': 'Token status refreshed',
                'token_status': token_status
            })
        
        # If tokens are exhausted, suggest making a request
        return jsonify({
            'success': False,
            'error': 'No tokens remaining. Please submit a token request to get more tokens.',
            'token_status': token_status,
            'suggest_request': True
        }), 400
        
    except Exception as e:
        print(f"Error in refresh tokens API: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

def validate_shriram_email(email):
    """Validate if email is from shrirampistons.com domain"""
    return email.lower().endswith('@shrirampistons.com')

@app.route('/api/chatbot-get-employee-status', methods=['GET'])
def chatbot_get_employee_status():
    """Get employee's current token status"""
    employee_email = request.args.get('email')
    
    if not employee_email:
        return jsonify({
            'success': False,
            'error': 'Employee email is required'
        }), 400
    
    if not validate_shriram_email(employee_email):
        return jsonify({
            'success': False,
            'error': 'Please use a valid @shrirampistons.com email address'
        }), 400
    
    token_status = get_user_token_status(employee_email)
    
    return jsonify({
        'success': True,
        'employee_email': employee_email,
        'current_usage': token_status['current_usage'],
        'total_limit': token_status['total_limit'],
        'remaining_tokens': token_status['remaining'],
        'status': token_status['status']
    })
    
@app.before_request
def clear_session_on_startup():
    """Clear session for protected routes if not authenticated"""
    protected_routes = ['chatbot', 'ask', 'history', 'feedback', 'tokens']
    
    if request.endpoint in protected_routes and 'user_email' not in session:
        if request.method == 'POST':
            return jsonify({"error": "Authentication required"}), 401
        return redirect(url_for('home'))

# ==================== FLASK ROUTES ====================

# Fixed Flask Routes
@app.route("/")
def home():
    """Main entry point - ALWAYS show login page first"""
    # Clear any existing session to force fresh login
    session.clear()
    print("🔐 Main route accessed - showing login page")
    return render_template('login.html')

@app.route("/login")
def login_page():
    """Alternative login route - also shows login page"""
    session.clear()
    print("🔐 Login route accessed - showing login page")
    return render_template('login.html')

@app.route("/authenticate", methods=["POST"])
def authenticate():
    """Process login credentials from synced users database"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400
            
        email = data.get("email", "").lower().strip()
        password = data.get("password", "").strip()
        
        print(f"🔐 Login attempt for: {email}")
        
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        
        # Clear any existing session first
        session.clear()
        
        # Check if sync is needed and sync if required
        if check_user_sync_needed():
            print("🔄 Syncing users from Excel...")
            sync_success = sync_users_excel_to_db()
            if sync_success:
                update_sync_flag()
            else:
                print("⚠️ User sync failed, but continuing with existing data")
        
        # Now authenticate from database (which is synced with Excel)
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        c.execute("SELECT email, password, name, department FROM users WHERE LOWER(email) = LOWER(?)", (email,))
        user_data = c.fetchone()
        
        if user_data and user_data[1] == password:
            # Login successful - set session
            session['user_email'] = user_data[0]
            session['user_name'] = user_data[2]
            session['user_department'] = user_data[3] or ''
            session.permanent = True
            
            # Update last login timestamp
            update_user_last_login(email)
            
            # Initialize user tokens if needed
            get_user_tokens(email)
            
            conn.close()
            
            print(f"✅ Login successful for: {user_data[2]} ({email})")
            
            return jsonify({
                "success": True,
                "message": f"Welcome {user_data[2]}!",
                "user_name": user_data[2],
                "redirect": "/chatbot"
            })
        else:
            conn.close()
            print(f"❌ Invalid credentials for: {email}")
            return jsonify({"error": "Invalid email or password"}), 401
            
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return jsonify({"error": "Login system error"}), 500
    
@app.route("/api/admin/sync-users", methods=["POST"])
def manual_sync_users():
    """Manually sync users from Excel to database"""
    try:
        print("🔄 Manual user sync initiated...")
        
        success = sync_users_excel_to_db()
        
        if success:
            update_sync_flag()
            return jsonify({
                "success": True,
                "message": "Users synchronized successfully from Excel file"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to sync users"
            }), 500
            
    except Exception as e:
        print(f"❌ Error in manual sync: {e}")
        return jsonify({"error": "Error syncing users"}), 500
    
@app.route("/api/admin/all-users", methods=["GET"])
def get_all_database_users():
    """Get all users from database (admin only)"""
    try:
        users = get_all_active_users()
        return jsonify({
            "users": users,
            "count": len(users),
            "message": f"Retrieved {len(users)} users from database"
        })
    except Exception as e:
        print(f"❌ Error getting all users: {e}")
        return jsonify({"error": "Error retrieving users"}), 500

@app.route("/debug-user-sync")
def debug_user_sync():
    """Debug route to check user synchronization status"""
    try:
        excel_path = "data/users.xlsx"
        sync_flag_path = "users_sync.flag"
        
        # Check Excel file status
        excel_exists = os.path.exists(excel_path)
        excel_mtime = os.path.getmtime(excel_path) if excel_exists else None
        
        # Check sync flag
        sync_flag_exists = os.path.exists(sync_flag_path)
        last_sync_time = None
        if sync_flag_exists:
            with open(sync_flag_path, 'r') as f:
                last_sync_time = float(f.read().strip())
        
        # Get user counts
        excel_users = load_users_from_excel(excel_path) if excel_exists else {}
        
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users")
        db_user_count = c.fetchone()[0]
        conn.close()
        
        return jsonify({
            "excel_file": {
                "exists": excel_exists,
                "modified_time": excel_mtime,
                "user_count": len(excel_users)
            },
            "database": {
                "user_count": db_user_count
            },
            "sync_status": {
                "flag_exists": sync_flag_exists,
                "last_sync_time": last_sync_time,
                "sync_needed": check_user_sync_needed()
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/chatbot")
def chatbot():
    """Chatbot page - only accessible after successful login"""
    if 'user_email' not in session:
        print("❌ Unauthorized chatbot access - redirecting to login")
        return redirect(url_for('home'))
    
    user_name = session.get('user_name', 'User')
    print(f"✅ Chatbot access granted for: {user_name}")
    return render_template('index.html')

@app.route("/logout")
def logout():
    """Logout and redirect to login page"""
    user_name = session.get('user_name', 'User')
    session.clear()
    print(f"👋 User logged out: {user_name}")
    return redirect(url_for('home'))

# ==================== NEW USER INFO API ROUTE ====================

@app.route("/user-info", methods=["GET"])
def get_user_info():
    """Get current user information - requires authentication"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    return jsonify({
        "user_name": session.get('user_name', 'User'),
        "user_email": session.get('user_email', ''),
        "user_department": session.get('user_department', ''),
        "authenticated": True
    })

# ==================== API ROUTES (Protected) ====================

@app.route("/ask", methods=["POST", "OPTIONS"])
def handle_ask():
    """Main ask endpoint - requires authentication"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    # Check authentication
    if 'user_email' not in session:
        return jsonify({"error": "Please login first"}), 401
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400
        
    question = data.get("query", "").strip()
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    # Use session user email
    user_id = session['user_email']

    try:
        # Get current token info
        new_limit, new_token_used, new_token_remaining = get_user_tokens(user_id)
        
        # Process the question
        answer = process_question(question, user_id)
        
        # Get updated token info
        _, new_token_used, new_token_remaining = get_user_tokens(user_id)
        
        return jsonify({
    "answer": answer,
    "tokens_used": new_token_used,
    "tokens_remaining": new_token_remaining,
    "token_limit": new_limit
})
    except Exception as e:
        print(f"❌ Error in handle_ask: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
# Add this route to your Flask app (around line 600-700 in your Python file):

# Replace your existing /delete-chat route with this fixed version:

@app.route("/api/user-updates", methods=["GET"])
def get_user_updates():
    """Get real-time updates for current user"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    user_email = session['user_email']
    last_check = request.args.get('last_check', (datetime.now() - timedelta(minutes=1)).isoformat())
    
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        updates = []
        
        # Check for recent token request updates
        c.execute("""SELECT id, tokens_requested, status, processed_at
                     FROM token_requests 
                     WHERE user_email = ? AND processed_at > ? AND status != 'pending'""", 
                  (user_email, last_check))
        
        for row in c.fetchall():
            updates.append({
                "type": "token_update",
                "message": f"Your token request for {row[1]} tokens was {row[2]}",
                "status": row[2],
                "timestamp": row[3]
            })
        
        # Check for recent feedback updates
        c.execute("""SELECT id, status, reviewed_at
                     FROM incorrect_feedback 
                     WHERE user_email = ? AND reviewed_at > ? AND status != 'pending'""", 
                  (user_email, last_check))
        
        for row in c.fetchall():
            updates.append({
                "type": "feedback_update",
                "message": f"Your feedback correction was {row[1]}",
                "status": row[1],
                "timestamp": row[2]
            })
        
        conn.close()
        
        # Get current token info
        token_limit, token_used, token_remaining = get_user_tokens(user_email)
        
        return jsonify({
            "updates": updates,
            "has_updates": len(updates) > 0,
            "current_tokens": {
                "limit": token_limit,
                "used": token_used,
                "remaining": token_remaining
            }
        })
        
    except Exception as e:
        print(f"❌ Error getting user updates: {e}")
        return jsonify({"error": "Error retrieving updates"}), 500

@app.route("/delete-chat", methods=["POST"])
def delete_chat():
    """Delete individual chat from history"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400
            
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Missing question"}), 400
        
        user_id = session['user_email']
        print(f"🗑️ Attempting to delete chat for user {user_id}: '{question[:50]}...'")
        
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        # First, check if the chat exists
        c.execute("SELECT COUNT(*) FROM chat_history WHERE user_id = ? AND question = ?", 
                  (user_id, question))
        count_before = c.fetchone()[0]
        print(f"📊 Found {count_before} matching chats")
        
        if count_before == 0:
            # Try partial match if exact match fails
            c.execute("SELECT question FROM chat_history WHERE user_id = ? AND question LIKE ?", 
                      (user_id, f"%{question}%"))
            similar_questions = c.fetchall()
            print(f"🔍 Found {len(similar_questions)} similar questions")
            
            if similar_questions:
                # Delete the first matching question
                actual_question = similar_questions[0][0]
                c.execute("DELETE FROM chat_history WHERE user_id = ? AND question = ?", 
                          (user_id, actual_question))
                deleted_count = c.rowcount
            else:
                conn.close()
                return jsonify({"error": "Chat not found"}), 404
        else:
            # Delete exact match
            c.execute("DELETE FROM chat_history WHERE user_id = ? AND question = ?", 
                      (user_id, question))
            deleted_count = c.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            print(f"✅ Successfully deleted {deleted_count} chat(s)")
            return jsonify({"message": "Chat deleted successfully", "deleted_count": deleted_count})
        else:
            print("❌ No chats were deleted")
            return jsonify({"error": "Failed to delete chat"}), 500
            
    except Exception as e:
        print(f"❌ Error deleting chat: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    
@app.route("/api/admin/sync-corrections", methods=["POST"])
def manual_sync_corrections():
    """Manually sync all approved corrections to Pinecone"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get all approved/implemented feedback
        c.execute("""SELECT original_query, original_response, corrected_answer
                     FROM incorrect_feedback 
                     WHERE status IN ('approved', 'implemented')""")
        
        corrections = c.fetchall()
        conn.close()
        
        if not corrections:
            return jsonify({"message": "No approved corrections found"})
        
        success_count = 0
        for original_query, original_response, corrected_answer in corrections:
            if update_pinecone_with_correction(original_query, original_response, corrected_answer):
                success_count += 1
        
        return jsonify({
            "message": f"Synced {success_count}/{len(corrections)} corrections to Pinecone",
            "total_corrections": len(corrections),
            "successful_syncs": success_count
        })
        
    except Exception as e:
        print(f"❌ Error syncing corrections: {e}")
        return jsonify({"error": "Error syncing corrections"}), 500    
    
# Add this route if it doesn't exist, or replace the existing one:

@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Clear all chat history for the current user"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    try:
        user_id = session['user_email']
        user_name = session.get('user_name', 'User')
        
        print(f"🗑️ Clearing all history for user: {user_name} ({user_id})")
        
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        # Count existing chats before deletion
        c.execute("SELECT COUNT(*) FROM chat_history WHERE user_id = ?", (user_id,))
        count_before = c.fetchone()[0]
        print(f"📊 Found {count_before} chats to delete")
        
        if count_before == 0:
            conn.close()
            return jsonify({"message": "No chat history to clear"})
        
        # Delete all chat history for this user
        c.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        deleted_count = c.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"✅ Successfully deleted {deleted_count} chat records")
        
        return jsonify({
            "message": f"Successfully cleared {deleted_count} chat records",
            "deleted_count": deleted_count
        })
        
    except Exception as e:
        print(f"❌ Error clearing history: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/save-login", methods=["POST", "OPTIONS"])
def save_login():
    """Handle user login"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    data = request.get_json()
    email = data.get("email")
    name = data.get("name")

    if not email or not name:
        return jsonify({"error": "Missing name or email"}), 400

    # Initialize user tokens if needed
    get_user_tokens(email)
    
    print(f"🔐 Login received: {name} ({email})")
    return jsonify({"status": "success", "message": f"Welcome {name}!"})

@app.route("/test-ask", methods=["POST"])
def handle_test_question():
    """Test question handling endpoint without authentication"""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    # Use a default test user
    test_email = "test-user@sprl.com"

    try:
        answer = process_question(question, test_email)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"❌ Error in handle_test_question: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/history", methods=["GET", "POST"])
def get_history():
    """Get chat history for authenticated user"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    user_id = session['user_email']
    
    try:
        history = get_chat_history(user_id, limit=20)
        formatted_history = []
        for question, answer in history:
            formatted_history.append([{"question": question, "answer": answer}])
        
        return jsonify({"history": formatted_history})
    except Exception as e:
        print(f"❌ Error getting history: {e}")
        return jsonify({"error": "Error retrieving history"}), 500

# Modified feedback route
@app.route("/feedback", methods=["POST"])
def handle_feedback():
    """Handle feedback and save incorrect responses to request.db for admin review"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    data = request.json
    user_email = session.get('user_email')
    user_name = session.get('user_name', 'Unknown User')
    user_department = session.get('user_department', 'Unknown Department')
    
    original_query = data.get("original_query", "")
    original_response = data.get("original_response", "")
    feedback_type = data.get("feedback_type")
    corrected_answer = data.get("corrected_answer", "")

    if not original_response or not feedback_type:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        print(f"📝 Feedback from {user_name}: {feedback_type}")

        if feedback_type == "correct":
            print("   ✅ User marked response as correct")
            return jsonify({"status": "Feedback recorded - Thank you!"}), 200

        elif feedback_type == "incorrect":
            if not corrected_answer:
                return jsonify({"error": "Corrected answer is required for incorrect feedback"}), 400

            print(f"   ❌ User provided correction: {corrected_answer[:100]}...")
            
            # Save to original database for immediate use (existing functionality)
            save_feedback(user_email, original_query, original_response, corrected_answer)
            
            # Save to request.db for admin review
            conn = sqlite3.connect("request.db")
            c = conn.cursor()
            
            # Create table if it doesn't exist
            c.execute("""CREATE TABLE IF NOT EXISTS incorrect_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL,
                user_email TEXT NOT NULL,
                user_department TEXT,
                original_query TEXT NOT NULL,
                original_response TEXT NOT NULL,
                corrected_answer TEXT NOT NULL,
                feedback_timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                reviewed_by TEXT,
                reviewed_at TEXT
            )""")
            
            feedback_timestamp = datetime.now().isoformat()
            
            c.execute("""INSERT INTO incorrect_feedback 
                         (user_name, user_email, user_department, original_query, 
                          original_response, corrected_answer, feedback_timestamp)
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                      (user_name, user_email, user_department, original_query, 
                       original_response, corrected_answer, feedback_timestamp))
            
            feedback_id = c.lastrowid
            conn.commit()
            conn.close()
            
            print(f"✅ Incorrect feedback saved to request.db: ID {feedback_id}")
            
            return jsonify({
                "status": "success",
                "message": "Thank you for your feedback! Your correction has been sent to administrators for review.",
                "feedback_id": feedback_id,
                "note": "The corrected answer is now available for your future queries, and administrators will review it for system-wide improvements."
            }), 200

        else:
            return jsonify({"error": "Invalid feedback type"}), 400

    except Exception as e:
        print(f"❌ Error processing feedback: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/tokens", methods=["POST"])
def get_user_token_info():
    """Get user token information - requires authentication"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    user_id = session['user_email']
    
    try:
        token_limit, token_used, token_remaining = get_user_tokens(user_id)
        return jsonify({
            "token_limit": token_limit,
            "token_used": token_used,
            "token_remaining": token_remaining
        })
    except Exception as e:
        print(f"❌ Error getting token info: {e}")
        return jsonify({"error": "Error retrieving token info"}), 500
    

# ==================== NEW API ENDPOINTS FOR CROSS-APP COMMUNICATION ====================

@app.route("/api/request-status/<int:request_id>", methods=["GET"])
def get_request_status(request_id):
    """Get status of a specific request"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    user_email = session['user_email']
    
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get token request status
        c.execute("""SELECT id, status, processed_by, processed_at, tokens_requested 
                     FROM token_requests 
                     WHERE id = ? AND user_email = ?""", (request_id, user_email))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            return jsonify({
                "request_id": result[0],
                "status": result[1],
                "processed_by": result[2],
                "processed_at": result[3],
                "tokens_requested": result[4]
            })
        else:
            return jsonify({"error": "Request not found"}), 404
            
    except Exception as e:
        print(f"❌ Error getting request status: {e}")
        return jsonify({"error": "Error retrieving request status"}), 500

@app.route("/api/my-requests", methods=["GET"])
def get_my_requests():
    """Get all requests for current user"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    user_email = session['user_email']
    
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get token requests
        c.execute("""SELECT id, tokens_requested, reason, priority, status, 
                            request_timestamp, processed_at
                     FROM token_requests 
                     WHERE user_email = ?
                     ORDER BY request_timestamp DESC""", (user_email,))
        
        token_requests = []
        for row in c.fetchall():
            token_requests.append({
                "id": row[0],
                "tokens_requested": row[1],
                "reason": row[2],
                "priority": row[3],
                "status": row[4],
                "submitted_at": row[5],
                "processed_at": row[6],
                "type": "token_request"
            })
        
        # Get feedback requests
        c.execute("""SELECT id, original_query, status, feedback_timestamp, reviewed_at
                     FROM incorrect_feedback 
                     WHERE user_email = ?
                     ORDER BY feedback_timestamp DESC""", (user_email,))
        
        feedback_requests = []
        for row in c.fetchall():
            feedback_requests.append({
                "id": row[0],
                "query": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                "status": row[2],
                "submitted_at": row[3],
                "reviewed_at": row[4],
                "type": "feedback"
            })
        
        conn.close()
        
        return jsonify({
            "token_requests": token_requests,
            "feedback_requests": feedback_requests,
            "total_requests": len(token_requests) + len(feedback_requests)
        })
        
    except Exception as e:
        print(f"❌ Error getting user requests: {e}")
        return jsonify({"error": "Error retrieving requests"}), 500

# ==================== NOTIFICATION SYSTEM ====================

@app.route("/api/notifications", methods=["GET"])
def get_notifications():
    """Get notifications for user about request updates"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    user_email = session['user_email']
    
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get recently processed requests (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        
        c.execute("""SELECT 'token' as type, id, tokens_requested, status, processed_at
                     FROM token_requests 
                     WHERE user_email = ? AND processed_at > ? AND status != 'pending'
                     UNION ALL
                     SELECT 'feedback' as type, id, 0, status, reviewed_at
                     FROM incorrect_feedback 
                     WHERE user_email = ? AND reviewed_at > ? AND status != 'pending'
                     ORDER BY processed_at DESC""", 
                  (user_email, week_ago, user_email, week_ago))
        
        notifications = []
        for row in c.fetchall():
            if row[0] == 'token':
                message = f"Your token request for {row[2]} tokens was {row[3]}"
            else:
                message = f"Your feedback correction was {row[3]}"
            
            notifications.append({
                "type": row[0],
                "id": row[1],
                "message": message,
                "status": row[3],
                "timestamp": row[4]
            })
        
        conn.close()
        
        return jsonify({
            "notifications": notifications,
            "count": len(notifications)
        })
        
    except Exception as e:
        print(f"❌ Error getting notifications: {e}")
        return jsonify({"error": "Error retrieving notifications"}), 500    

# ==================== UTILITY ROUTES ====================

@app.route("/api/status")
def api_status():
    """API status endpoint"""
    return jsonify({
        "status": "running",
        "message": "SPRL Chatbot backend is running",
        "login_required": True
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# ==================== DEBUG ROUTE (Remove in production) ====================

@app.route("/debug-session")
def debug_session():
    """Debug route to check session status"""
    return jsonify({
        "session_active": 'user_email' in session,
        "user_email": session.get('user_email', 'Not logged in'),
        "user_name": session.get('user_name', 'Not logged in'),
        "session_keys": list(session.keys())
    })

@app.route("/debug-users")
def debug_users():
    """Debug route to check users from Excel"""
    users = load_users_from_excel("data/users.xlsx")
    return jsonify({
        "users_count": len(users),
        "user_emails": list(users.keys()),
        "sample_user": {k: {**v, "password": "***"} for k, v in list(users.items())[:1]}
    })

# ==================== NEW API ROUTES FOR TOKEN MANAGEMENT ====================

@app.route("/api/admin/users", methods=["GET"])
def get_all_users():
    """Get all users with token information"""
    try:
        users_data = get_all_users_token_info()
        return jsonify({"users": users_data})
    except Exception as e:
        print(f"❌ Error getting users: {e}")
        return jsonify({"error": "Error retrieving users"}), 500
    
@app.route("/grant-tokens", methods=["POST"])
def grant_user_tokens():
    """Grant additional tokens to current user"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    data = request.get_json()
    additional_tokens = data.get("tokens", 0)
    user_id = session['user_email']
    
    try:
        token_limit, token_used, token_remaining = get_user_tokens(user_id)
        new_limit = token_limit + additional_tokens
        update_user_token_limit(user_id, new_limit)
        
        _, _, new_remaining = get_user_tokens(user_id)
        
        return jsonify({
            "message": f"Granted {additional_tokens} tokens",
            "token_limit": new_limit,
            "token_used": token_used,
            "token_remaining": new_remaining
        })
    except Exception as e:
        return jsonify({"error": "Error granting tokens"}), 500    

@app.route("/api/admin/grant-tokens", methods=["POST"])
def grant_tokens():
    """Grant additional tokens to a user"""
    try:
        data = request.get_json()
        user_email = data.get("user_email")
        additional_tokens = int(data.get("additional_tokens", 0))
        
        if not user_email or additional_tokens <= 0:
            return jsonify({"error": "Invalid user or token amount"}), 400
        
        # Get current token info
        token_limit, token_used, token_remaining = get_user_tokens(user_email)
        new_limit = token_limit + additional_tokens
        
        # Update token limit
        update_user_token_limit(user_email, new_limit)
        
        return jsonify({
            "message": f"Granted {additional_tokens} tokens to {user_email}",
            "new_limit": new_limit,
            "new_remaining": new_limit - token_used
        })
        
    except Exception as e:
        print(f"❌ Error granting tokens: {e}")
        return jsonify({"error": "Error granting tokens"}), 500

# New admin API routes for request management
@app.route("/api/admin/token-requests", methods=["GET"])
def get_admin_token_requests():
    """Get all pending token requests for admin portal"""
    try:
        requests = get_pending_token_requests()
        return jsonify({"requests": requests, "count": len(requests)})
    except Exception as e:
        print(f"❌ Error getting token requests: {e}")
        return jsonify({"error": "Error retrieving requests"}), 500

@app.route("/api/admin/feedback-requests", methods=["GET"])
def get_admin_feedback_requests():
    """Get all pending feedback requests for admin portal"""
    try:
        feedback_list = get_pending_feedback_requests()
        return jsonify({"feedback": feedback_list, "count": len(feedback_list)})
    except Exception as e:
        print(f"❌ Error getting feedback requests: {e}")
        return jsonify({"error": "Error retrieving feedback"}), 500

@app.route("/api/admin/approve-token-request", methods=["POST"])
def approve_token_request():
    """Approve token request and grant tokens"""
    try:
        data = request.get_json()
        request_id = int(data.get("request_id"))
        admin_email = data.get("admin_email", "admin@system.com")
        tokens_to_grant = int(data.get("tokens_granted", 0))
        
        if not request_id or tokens_to_grant <= 0:
            return jsonify({"error": "Invalid request ID or token amount"}), 400
        
        success, message = update_token_request_status(
            request_id=request_id,
            status="approved",
            admin_email=admin_email,
            tokens_granted=tokens_to_grant
        )
        
        if success:
            return jsonify({"message": f"Token request approved and {tokens_to_grant} tokens granted"})
        else:
            return jsonify({"error": message}), 500
            
    except Exception as e:
        print(f"❌ Error approving token request: {e}")
        return jsonify({"error": "Error processing approval"}), 500

@app.route("/api/admin/approve-feedback", methods=["POST"])
def approve_feedback_with_pinecone():
    """Approve feedback and automatically update Pinecone"""
    try:
        data = request.get_json()
        feedback_id = int(data.get("feedback_id"))
        admin_email = data.get("admin_email", "admin@system.com")
        
        if not feedback_id:
            return jsonify({"error": "Invalid feedback ID"}), 400
        
        # Get feedback details before updating
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        c.execute("""SELECT user_email, original_query, original_response, corrected_answer
                     FROM incorrect_feedback WHERE id = ?""", (feedback_id,))
        feedback_data = c.fetchone()
        
        if not feedback_data:
            conn.close()
            return jsonify({"error": "Feedback not found"}), 404
        
        user_email, original_query, original_response, corrected_answer = feedback_data
        
        # Update feedback status to 'approved'
        reviewed_at = datetime.now().isoformat()
        c.execute("""UPDATE incorrect_feedback 
                     SET status = 'approved', reviewed_by = ?, reviewed_at = ?
                     WHERE id = ?""",
                  (admin_email, reviewed_at, feedback_id))
        
        conn.commit()
        conn.close()
        
        # Update Pinecone with the correction
        print(f"🔄 Updating Pinecone with approved correction...")
        pinecone_success = update_pinecone_with_correction(original_query, original_response, corrected_answer)
        
        # Also update local users.db for immediate use
        if pinecone_success:
            conn_users = sqlite3.connect("users.db")
            c_users = conn_users.cursor()
            c_users.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?)", 
                          (user_email, original_query, original_response, corrected_answer))
            conn_users.commit()
            conn_users.close()
            
            # Log admin action
            log_admin_action(admin_email, "feedback_approved", user_email, 
                           f"Approved feedback ID: {feedback_id} and updated Pinecone")
            
            return jsonify({
                "success": True,
                "message": "Feedback approved and Pinecone updated successfully",
                "pinecone_updated": True,
                "feedback_id": feedback_id
            })
        else:
            return jsonify({
                "success": False,
                "message": "Feedback approved but Pinecone update failed",
                "pinecone_updated": False,
                "feedback_id": feedback_id
            }), 500
            
    except Exception as e:
        print(f"❌ Error approving feedback: {e}")
        return jsonify({"error": "Error processing approval"}), 500

@app.route("/api/admin/approve-token-request-enhanced", methods=["POST"])
def approve_token_request_enhanced():
    """Enhanced token approval with real-time updates"""
    try:
        data = request.get_json()
        request_id = int(data.get("request_id"))
        admin_email = data.get("admin_email", "admin@system.com")
        tokens_to_grant = int(data.get("tokens_granted", 0))
        
        if not request_id or tokens_to_grant <= 0:
            return jsonify({"error": "Invalid request ID or token amount"}), 400
        
        # Get request details
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        c.execute("SELECT user_email, user_name FROM token_requests WHERE id = ?", (request_id,))
        request_data = c.fetchone()
        
        if not request_data:
            conn.close()
            return jsonify({"error": "Request not found"}), 404
        
        user_email, user_name = request_data
        processed_at = datetime.now().isoformat()
        
        # Update request status
        c.execute("""UPDATE token_requests 
                     SET status = 'approved', processed_by = ?, processed_at = ?
                     WHERE id = ?""",
                  (admin_email, processed_at, request_id))
        
        conn.commit()
        conn.close()
        
        # Grant tokens to user
        _, _, new_token_remaining = get_user_tokens(user_email)
        token_limit, token_used, token_remaining = get_user_tokens(user_email)
        new_limit = token_limit + tokens_to_grant
        update_user_token_limit(user_email, new_limit)
        
        # Get updated token info
        _, _, new_token_remaining = get_user_tokens(user_email)
        
        # Log admin action
        log_admin_action(admin_email, "token_grant", user_email, 
                       f"Granted {tokens_to_grant} tokens (Request ID: {request_id})")
        
        return jsonify({
            "success": True,
            "message": f"Approved and granted {tokens_to_grant} tokens to {user_name}",
            "request_id": request_id,
            "user_email": user_email,
            "tokens_granted": tokens_to_grant,
            "new_token_limit": new_limit,
            "new_token_remaining": new_token_remaining
        })
        
    except Exception as e:
        print(f"❌ Error approving token request: {e}")
        return jsonify({"error": "Error processing approval"}), 500



@app.route("/api/admin/reject-token-request", methods=["POST"])
def reject_token_request():
    """Reject token request"""
    try:
        data = request.get_json()
        request_id = int(data.get("request_id"))
        admin_email = data.get("admin_email", "admin@system.com")
        
        if not request_id:
            return jsonify({"error": "Invalid request ID"}), 400
        
        success, message = update_token_request_status(
            request_id=request_id,
            status="rejected",
            admin_email=admin_email
        )
        
        if success:
            return jsonify({"message": "Token request rejected"})
        else:
            return jsonify({"error": message}), 500
            
    except Exception as e:
        print(f"❌ Error rejecting token request: {e}")
        return jsonify({"error": "Error processing rejection"}), 500

@app.route("/api/admin/review-feedback", methods=["POST"])
def review_feedback():
    """Mark feedback as reviewed"""
    try:
        data = request.get_json()
        feedback_id = int(data.get("feedback_id"))
        admin_email = data.get("admin_email", "admin@system.com")
        status = data.get("status", "reviewed")  # reviewed, implemented, etc.
        
        if not feedback_id:
            return jsonify({"error": "Invalid feedback ID"}), 400
        
        success, message = update_feedback_status(
            feedback_id=feedback_id,
            status=status,
            admin_email=admin_email
        )
        
        if success:
            return jsonify({"message": f"Feedback marked as {status}"})
        else:
            return jsonify({"error": message}), 500
            
    except Exception as e:
        print(f"❌ Error reviewing feedback: {e}")
        return jsonify({"error": "Error processing review"}), 500

@app.route("/api/admin/dashboard-stats", methods=["GET"])
def get_admin_dashboard_stats():
    """Get statistics for admin dashboard"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Count pending requests
        c.execute("SELECT COUNT(*) FROM token_requests WHERE status = 'pending'")
        pending_token_requests = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM incorrect_feedback WHERE status = 'pending'")
        pending_feedback = c.fetchone()[0]
        
        # Count total requests this month
        current_month = datetime.now().strftime("%Y-%m")
        c.execute("SELECT COUNT(*) FROM token_requests WHERE created_at LIKE ?", (f"{current_month}%",))
        monthly_requests = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM incorrect_feedback WHERE created_at LIKE ?", (f"{current_month}%",))
        monthly_feedback = c.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            "pending_token_requests": pending_token_requests,
            "pending_feedback": pending_feedback,
            "monthly_token_requests": monthly_requests,
            "monthly_feedback": monthly_feedback,
            "total_pending": pending_token_requests + pending_feedback
        })
        
    except Exception as e:
        print(f"❌ Error getting dashboard stats: {e}")
        return jsonify({"error": "Error retrieving statistics"}), 500
    
@app.route("/recharge-tokens", methods=["POST"])
def recharge_tokens():
    """Save token request to request.db - UPDATED for standardized schema"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    data = request.get_json()
    additional_tokens = int(data.get("tokens", 0))
    reason = data.get("reason", "").strip()
    priority = data.get("priority", "low")
    
    user_email = session['user_email']
    user_name = session.get('user_name', 'Unknown User')
    user_department = session.get('user_department', 'Unknown Department')
    
    if additional_tokens <= 0:
        return jsonify({"error": "Invalid token amount"}), 400
    
    if not reason:
        return jsonify({"error": "Reason is required"}), 400
    
    try:
        # Get current token status
        token_limit, token_used, token_remaining = get_user_tokens(user_email)
        token_remaining_calc = token_limit - token_used

        # Get the latest new_token_limit from previous requests
        conn_check = sqlite3.connect("request.db")
        c_check = conn_check.cursor()
        c_check.execute("""SELECT new_token_limit FROM token_requests 
                        WHERE user_email = ? 
                        ORDER BY created_at DESC LIMIT 1""", (user_email,))
        last_request = c_check.fetchone()
        conn_check.close()

        # Use previous new_token_limit or current token_limit if no previous requests
        current_token_limit = last_request[0] if last_request else token_limit
        token_remaining_calc = current_token_limit - token_used
        new_token_limit = current_token_limit + additional_tokens
        new_token_remaining = token_remaining_calc + additional_tokens
        # Connect to request.db and save the token request
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        request_timestamp = datetime.now().isoformat()
        
        # UPDATED: Using standardized column names
        c.execute("""INSERT INTO token_requests 
            (user_name, user_email, user_department, tokens_requested, 
            current_tokens_used, current_token_limit, token_remaining,
            new_token_limit, new_token_remaining, reason, priority, request_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_name, user_email, user_department, additional_tokens, 
        token_used, current_token_limit, token_remaining_calc, new_token_limit,  # CHANGED
        new_token_remaining, reason, priority, request_timestamp))


        
        request_id = c.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Token request saved to request.db: ID {request_id} for {user_name}")
        
        return jsonify({
            "success": True,
            "message": f"Token request submitted successfully! Request ID: {request_id}",
            "request_id": request_id,
            "status": "pending",
            "note": "Your request has been sent to administrators for approval. You will be notified once processed.",
            "current_tokens": {
                "used": token_used,
                "limit": token_limit,
                "remaining": token_remaining
            }
        })
        
    except Exception as e:
        print(f"❌ Error submitting token request: {e}")
        return jsonify({"error": "Error submitting request. Please try again."}), 500
@app.route("/api/admin/handle-request", methods=["POST"])
def handle_token_request():
    """Handle token request (approve/reject/hold)"""
    try:
        data = request.get_json()
        request_id = data.get("request_id")
        action = data.get("action")  # accept, reject, hold
        
        if action == "accept":
            # Logic to approve and grant tokens
            return jsonify({"message": "Request approved"})
        elif action == "reject":
            return jsonify({"message": "Request rejected"})
        elif action == "hold":
            return jsonify({"message": "Request put on hold"})
        else:
            return jsonify({"error": "Invalid action"}), 400
            
    except Exception as e:
        print(f"❌ Error handling request: {e}")
        return jsonify({"error": "Error processing request"}), 500

@app.route("/admin")
def admin_dashboard():
    """Admin dashboard page"""
    # Add simple admin authentication here if needed
    return render_template('admin.html')  # Your token management HTML

# ==================== DEBUGGING ROUTES FOR REQUEST DATABASE ====================
@app.route("/api/check-updates", methods=["GET"])
def check_for_updates():
    """Check for updates to user's requests and tokens"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    user_email = session['user_email']
    
    try:
        # Check token updates
        token_limit, token_used, token_remaining = get_user_tokens(user_email)
        
        # Check recent request updates (last 5 minutes)
        recent_time = (datetime.now() - timedelta(minutes=5)).isoformat()
        
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Check for recently processed token requests
        c.execute("""SELECT id, tokens_requested, status, processed_at
                     FROM token_requests 
                     WHERE user_email = ? AND processed_at > ? AND status != 'pending'""", 
                  (user_email, recent_time))
        
        token_updates = []
        for row in c.fetchall():
            token_updates.append({
                "type": "token_request",
                "id": row[0],
                "tokens_requested": row[1],
                "status": row[2],
                "processed_at": row[3]
            })
        
        # Check for recently reviewed feedback
        c.execute("""SELECT id, original_query, status, reviewed_at
                     FROM incorrect_feedback 
                     WHERE user_email = ? AND reviewed_at > ? AND status != 'pending'""", 
                  (user_email, recent_time))
        
        feedback_updates = []
        for row in c.fetchall():
            feedback_updates.append({
                "type": "feedback",
                "id": row[0],
                "query_preview": row[1][:50] + "..." if len(row[1]) > 50 else row[1],
                "status": row[2],
                "reviewed_at": row[3]
            })
        
        conn.close()
        
        return jsonify({
            "token_info": {
                "limit": token_limit,
                "used": token_used,
                "remaining": token_remaining
            },
            "updates": {
                "token_requests": token_updates,
                "feedback": feedback_updates,
                "has_updates": len(token_updates) > 0 or len(feedback_updates) > 0
            }
        })
        
    except Exception as e:
        print(f"❌ Error checking updates: {e}")
        return jsonify({"error": "Error checking for updates"}), 500

@app.route("/api/poll-updates", methods=["GET"])
def poll_updates():
    """Polling endpoint for real-time updates (called periodically by frontend)"""
    if 'user_email' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    user_email = session['user_email']
    last_check = request.args.get('last_check')  # ISO timestamp
    
    try:
        # If no last_check provided, use last 1 minute
        if not last_check:
            last_check = (datetime.now() - timedelta(minutes=1)).isoformat()
        
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get recent updates since last_check
        updates = []
        
        # Check token request updates
        c.execute("""SELECT id, tokens_requested, status, processed_at, processed_by
                     FROM token_requests 
                     WHERE user_email = ? AND processed_at > ? AND status != 'pending'
                     ORDER BY processed_at DESC""", 
                  (user_email, last_check))
        
        for row in c.fetchall():
            updates.append({
                "type": "token_request_update",
                "id": row[0],
                "tokens_requested": row[1],
                "status": row[2],
                "processed_at": row[3],
                "processed_by": row[4],
                "message": f"Your token request for {row[1]} tokens was {row[2]}"
            })
        
        # Check feedback updates
        c.execute("""SELECT id, original_query, status, reviewed_at, reviewed_by
                     FROM incorrect_feedback 
                     WHERE user_email = ? AND reviewed_at > ? AND status != 'pending'
                     ORDER BY reviewed_at DESC""", 
                  (user_email, last_check))
        
        for row in c.fetchall():
            updates.append({
                "type": "feedback_update",
                "id": row[0],
                "query_preview": row[1][:50] + "..." if len(row[1]) > 50 else row[1],
                "status": row[2],
                "reviewed_at": row[3],
                "reviewed_by": row[4],
                "message": f"Your feedback correction was {row[2]}"
            })
        
        conn.close()
        
        # Get current token info
        token_limit, token_used, token_remaining = get_user_tokens(user_email)
        
        return jsonify({
            "updates": updates,
            "has_new_updates": len(updates) > 0,
            "current_tokens": {
                "limit": token_limit,
                "used": token_used,
                "remaining": token_remaining
            },
            "last_check": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error polling updates: {e}")
        return jsonify({"error": "Error polling for updates"}), 500

# ==================== BATCH SYNC ENDPOINT ====================

@app.route("/api/admin/batch-sync", methods=["POST"])
def batch_sync_all():
    """Batch sync all approved changes to their respective systems"""
    try:
        data = request.get_json()
        admin_email = data.get("admin_email", "admin@system.com")
        
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        results = {
            "corrections_synced": 0,
            "tokens_granted": 0,
            "errors": []
        }
        
        # Sync all approved feedback to Pinecone
        c.execute("""SELECT id, original_query, original_response, corrected_answer, user_email
                     FROM incorrect_feedback 
                     WHERE status = 'approved'""")
        
        feedback_items = c.fetchall()
        for feedback_id, original_query, original_response, corrected_answer, user_email in feedback_items:
            try:
                if update_pinecone_with_correction(original_query, original_response, corrected_answer):
                    # Update status to implemented
                    c.execute("UPDATE incorrect_feedback SET status = 'implemented' WHERE id = ?", (feedback_id,))
                    results["corrections_synced"] += 1
                else:
                    results["errors"].append(f"Failed to sync feedback ID {feedback_id}")
            except Exception as e:
                results["errors"].append(f"Error syncing feedback ID {feedback_id}: {str(e)}")
        
        # Process all approved token requests
        c.execute("""SELECT id, user_email, tokens_requested
                     FROM token_requests 
                     WHERE status = 'approved'""")
        
        token_requests = c.fetchall()
        for request_id, user_email, tokens_requested in token_requests:
            try:
                # Grant tokens
                token_limit, token_used, token_remaining = get_user_tokens(user_email)
                new_limit = token_limit + tokens_requested
                update_user_token_limit(user_email, new_limit)
                
                # Update status to processed
                processed_at = datetime.now().isoformat()
                c.execute("""UPDATE token_requests 
                             SET status = 'processed', processed_by = ?, processed_at = ?
                             WHERE id = ?""",
                          (admin_email, processed_at, request_id))
                
                results["tokens_granted"] += tokens_requested
                
            except Exception as e:
                results["errors"].append(f"Error processing token request ID {request_id}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        # Log batch sync action
        log_admin_action(admin_email, "batch_sync", None, 
                       f"Synced {results['corrections_synced']} corrections, granted {results['tokens_granted']} tokens")
        
        return jsonify({
            "success": True,
            "message": "Batch sync completed",
            "results": results
        })
        
    except Exception as e:
        print(f"❌ Error in batch sync: {e}")
        return jsonify({"error": "Error in batch sync"}), 500


@app.route("/api/force-refresh-corrections", methods=["POST"])
def force_refresh_corrections():
    """Force refresh corrections from request.db to Pinecone"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get all approved corrections that might need syncing
        c.execute("""SELECT id, original_query, original_response, corrected_answer, user_email
                     FROM incorrect_feedback 
                     WHERE status IN ('approved', 'implemented')""")
        
        corrections = c.fetchall()
        conn.close()
        
        if not corrections:
            return jsonify({"message": "No approved corrections found", "synced": 0})
        
        # Sync each correction to Pinecone
        success_count = 0
        for correction_id, original_query, original_response, corrected_answer, user_email in corrections:
            # Update Pinecone
            if update_pinecone_with_correction(original_query, original_response, corrected_answer):
                # Also update local users.db
                conn_users = sqlite3.connect("users.db")
                c_users = conn_users.cursor()
                c_users.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?)", 
                              (user_email, original_query, original_response, corrected_answer))
                conn_users.commit()
                conn_users.close()
                success_count += 1
        
        return jsonify({
            "message": f"Synced {success_count}/{len(corrections)} corrections",
            "total_corrections": len(corrections),
            "synced": success_count
        })
        
    except Exception as e:
        print(f"❌ Error force refreshing corrections: {e}")
        return jsonify({"error": "Error syncing corrections"}), 500

@app.route("/debug-requests")
def debug_requests():
    """Debug route to check request database contents"""
    try:
        conn = sqlite3.connect("request.db")
        c = conn.cursor()
        
        # Get token requests
        c.execute("SELECT COUNT(*) FROM token_requests")
        token_count = c.fetchone()[0]
        
        c.execute("SELECT * FROM token_requests ORDER BY created_at DESC LIMIT 5")
        recent_tokens = c.fetchall()
        
        # Get feedback requests
        c.execute("SELECT COUNT(*) FROM incorrect_feedback")
        feedback_count = c.fetchone()[0]
        
        c.execute("SELECT * FROM incorrect_feedback ORDER BY created_at DESC LIMIT 5")
        recent_feedback = c.fetchall()
        
        conn.close()
        
        return jsonify({
            "token_requests_count": token_count,
            "feedback_requests_count": feedback_count,
            "recent_token_requests": recent_tokens,
            "recent_feedback": recent_feedback,
            "database_status": "connected"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "database_status": "error"
        })
# ==================== DATA UPLOAD FUNCTIONS ====================

def upload_chunks_to_pinecone(pkl_path="chunks.pkl", flag_file="upload_done.flag"):
    """Upload document chunks to Pinecone (one-time setup)"""
    if os.path.exists(flag_file):
        print("✅ Chunks already uploaded to Pinecone. Skipping re-upload.")
        return
    
    if not os.path.exists(pkl_path):
        print(f"⚠ Missing: {pkl_path}. Skipping chunk upload.")
        return

    try:
        with open(pkl_path, "rb") as f:
            chunks = pickle.load(f)

        index, _ = initialize_pinecone()
        if not index:
            return

        vectors = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("full_text", "").strip()
            if not text:
                continue

            embedding = embedding_model.embed_query(text)
            vector_id = f"chunk-{i}"
            metadata = {
                "source": "docx_chunk",
                "type": chunk.get("type", "unknown"),
                "text": text
            }
            vectors.append((vector_id, embedding, metadata))

        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc="📤 Uploading chunks to Pinecone"):
            batch = vectors[i:i + batch_size]
            index.upsert(batch)

        with open(flag_file, "w") as f:
            f.write("upload complete")
        print(f"✅ Uploaded {len(vectors)} chunks to Pinecone.")
        
    except Exception as e:
        print(f"❌ Error uploading chunks: {e}")
        


def upload_travel_records_to_pinecone(json_path="travel_records.json", flag_file="upload_travel_records_done.flag"):
    if os.path.exists(flag_file):
        print("✅ Travel records already uploaded to Pinecone. Skipping re-upload.")
        return

    try:
        with open(json_path, "r") as f:
            travel_data = json.load(f)

        records = travel_data.get("Travel Request", [])
        if not records:
            print("⚠ No travel records found.")
            return

        index, _ = initialize_pinecone()
        if not index:
            return

        vectors = []
        for i, record in enumerate(records):
            def safe_str(value):
                return str(value).strip() if value is not None else ""

            employee_name = safe_str(record.get("employee")).strip().lower()

            text = (
                f"Travel Request:\n"
                f"Employee: {employee_name.title()}\n"
                f"Department: {record.get('department')}\n"
                f"From: {record.get('origin')} to {record.get('destination')}\n"
                f"Date: {record.get('date')}\n"
                f"Email: {record.get('email')}\n"
                f"Cost: {record.get('cost')}"
            )

            metadata = {
                "source": "travel_json",
                "type": "travel",
                "employee": employee_name,
                "origin": safe_str(record.get("origin")),
                "department": safe_str(record.get("department")),
                "destination": safe_str(record.get("destination")),
                "date": safe_str(record.get("date")),
                "email": safe_str(record.get("email")),
                "cost": float(record.get("cost", 0))
            }

            embedding = embedding_model.embed_query(text)
            vector_id = f"travel-{i}"
            metadata["text"] = text
            vectors.append((vector_id, embedding, metadata))

        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc="📤 Uploading travel records"):
            batch = vectors[i:i + batch_size]
            index.upsert(batch)

        print(f"✅ Uploaded {len(vectors)} travel records to Pinecone.")
        
    except Exception as e:
        print(f"❌ Error uploading travel records: {e}")
        
     # Add this at the end:
    with open(flag_file, "w") as f:
        f.write("travel records upload complete")
    print(f"🚩 Created flag file: {flag_file}")
    
def check_pinecone_stats():
    """Check current stats of Pinecone index"""
    try:
        index, _ = initialize_pinecone()
        if index:
            stats = index.describe_index_stats()
            print(f"📊 Pinecone Index Stats:")
            print(f"   Total vectors: {stats.total_vector_count}")
            print(f"   Index fullness: {stats.index_fullness}")
    except Exception as e:
        print(f"⚠️ Could not fetch Pinecone stats: {e}")

def force_reupload_travel_records():
    """Force re-upload of travel records (for debugging)"""
    flag_files = [
        "upload_travel_records_done.flag",
        "upload_done.flag"  # For chunks
    ]
    for flag_file in flag_files:
        if os.path.exists(flag_file):
            os.remove(flag_file)
            print(f"🗑️ Removed flag file: {flag_file}")
    
    upload_travel_records_to_pinecone()
    upload_chunks_to_pinecone()

# ==================== INITIALIZATION ====================


if __name__ == "__main__":
    print("🚀 Initializing SPRL Chatbot with Pinecone + Gemini...")
    
    # Initialize databases
    init_db()
    init_request_db()
    migrate_token_requests_table()
    
    # Sync users from Excel to database
    print("👥 Syncing users from Excel to database...")
    if check_user_sync_needed():
        sync_success = sync_users_excel_to_db()
        if sync_success:
            update_sync_flag()
            print("✅ Users synced successfully")
        else:
            print("⚠️ User sync failed - please check users.xlsx file")
    else:
        print("✅ Users already synced")
    
    # Check Pinecone status
    check_pinecone_stats()
    
    # Upload data with flag protection
    upload_chunks_to_pinecone()
    upload_travel_records_to_pinecone()
    
    print("🚀 SPRL Chatbot is live!")
    print("📝 Note: Users are automatically synced from users.xlsx")
    print("   - Add/modify users in users.xlsx and restart the app")
    print("   - Or use /api/admin/sync-users endpoint for manual sync")
    
    app.run(host="0.0.0.0", port=5001, debug=True)