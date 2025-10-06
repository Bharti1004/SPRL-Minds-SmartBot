from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3
import os
from datetime import datetime, timedelta
import secrets
import smtplib
import pandas as pd
import hashlib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import re


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.permanent_session_lifetime = timedelta(hours=24)

# Database setup
ADMINS_EXCEL_FILE = 'data/admins.xlsx'
DATABASE = 'users.db'
REQUEST_DATABASE = 'request.db'  # Your request database
ADMIN_DATABASE = 'admins.db'    # For admin users  

def get_file_hash(filepath):
    """Get MD5 hash of file to detect changes"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return None

def create_default_excel_file():
    """Create default admins.xlsx file if it doesn't exist"""
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(ADMINS_EXCEL_FILE):
        print(f"📁 Creating empty {ADMINS_EXCEL_FILE} file...")
        
        # Create empty DataFrame with just the column headers
        empty_admins = pd.DataFrame(columns=[
            'name', 
            'email', 
            'department', 
            'password', 
            'is_active'
        ])
        
        empty_admins.to_excel(ADMINS_EXCEL_FILE, index=False)
        print(f"✅ Created empty {ADMINS_EXCEL_FILE} with column headers only")
        print("ℹ️  Please add admin details manually to the Excel file")

def sync_admins_from_excel():
    """Sync admins from Excel file to ADMIN database"""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        if not os.path.exists(ADMINS_EXCEL_FILE):
            create_default_excel_file()
            
        # Read Excel file
        df = pd.read_excel(ADMINS_EXCEL_FILE)
        
        # Validate required columns
        required_columns = ['name', 'email', 'department', 'password']
        if not all(col in df.columns for col in required_columns):
            print("❌ Excel file missing required columns: name, email, department, password")
            return False
        
        conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED: Use ADMIN_DATABASE
        cursor = conn.cursor()
        
        # Get current admins in database
        cursor.execute('SELECT email FROM admins')
        db_emails = {row[0] for row in cursor.fetchall()}
        
        # Get Excel admins
        excel_emails = set(df['email'].str.lower().str.strip())
        
        # Add/Update admins from Excel
        for _, row in df.iterrows():
            email = str(row['email']).lower().strip()
            name = str(row['name']).strip()
            department = str(row['department']).strip()
            password = str(row['password']).strip()
            is_active = int(row.get('is_active', 1))
            
            if not validate_shriram_email(email):
                print(f"⚠️  Skipping invalid email: {email}")
                continue
            
            password_hash = generate_password_hash(password)
            
            if email in db_emails:
                # Update existing admin
                cursor.execute('''
                    UPDATE admins 
                    SET name = ?, department = ?, password_hash = ?, is_active = ?
                    WHERE email = ?
                ''', (name, department, password_hash, is_active, email))
                print(f"🔄 Updated admin: {email}")
            else:
                # Add new admin
                cursor.execute('''
                    INSERT INTO admins (email, password_hash, name, department, is_active)
                    VALUES (?, ?, ?, ?, ?)
                ''', (email, password_hash, name, department, is_active))
                print(f"✅ Added new admin: {email}")
        
        # Remove admins not in Excel
        admins_to_remove = db_emails - excel_emails
        for email in admins_to_remove:
            cursor.execute('DELETE FROM admins WHERE email = ?', (email,))
            print(f"🗑️  Removed admin: {email}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Successfully synced {len(excel_emails)} admins from Excel")
        return True
        
    except Exception as e:
        print(f"❌ Error syncing admins from Excel: {e}")
        return False

def export_admins_to_excel():
    """Export current admins from ADMIN database to Excel file"""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED: Use ADMIN_DATABASE
        
        # Read admins from database (without password hashes)
        df = pd.read_sql_query('''
            SELECT name, email, department, is_active, 
                   'password123' as password,
                   created_at, last_login
            FROM admins 
            ORDER BY created_at
        ''', conn)
        
        conn.close()
        
        # Save to Excel
        df.to_excel(ADMINS_EXCEL_FILE, index=False)
        print(f"✅ Exported {len(df)} admins to {ADMINS_EXCEL_FILE}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting admins to Excel: {e}")
        return False

def init_db():
    """Initialize both databases with respective tables"""
    
    # Initialize USERS database (users.db)
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            last_login DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            password_hash TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_sync (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            last_hash TEXT,
            last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Initialize ADMINS database (admins.db)
    conn = sqlite3.connect(ADMIN_DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            code TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            used BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_sync (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            last_hash TEXT,
            last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Sync admins from Excel file
    print("🔄 Syncing admins from Excel file...")
    sync_admins_from_excel()


def get_user_by_email(email):
    """Get admin user from ADMIN database by email"""
    conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, email, password_hash, name, department, last_login, is_active
        FROM admins WHERE email = ? AND is_active = 1
    ''', (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {
            'id': user[0],
            'email': user[1],
            'password_hash': user[2],
            'name': user[3],
            'department': user[4],
            'last_login': user[5],
            'is_active': user[6]
        }
    return None

def update_last_login(user_id):
    """Update admin's last login timestamp"""
    conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE admins SET last_login = CURRENT_TIMESTAMP WHERE id = ?
    ''', (user_id,))
    conn.commit()
    conn.close()

def validate_shriram_email(email):
    """Validate if email belongs to shrirampistons.com domain"""
    pattern = r'^[a-zA-Z0-9._%+-]+@shrirampistons\.com$'
    return re.match(pattern, email) is not None

def store_verification_code(email, code):
    """Store verification code in ADMIN database"""
    conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
    cursor = conn.cursor()
    
    # Delete any existing unused codes for this email
    cursor.execute('DELETE FROM verification_codes WHERE email = ? AND used = 0', (email,))
    
    # Store new code with 15 minutes expiration
    expires_at = datetime.now() + timedelta(minutes=15)
    cursor.execute('''
        INSERT INTO verification_codes (email, code, expires_at)
        VALUES (?, ?, ?)
    ''', (email, code, expires_at))
    
    conn.commit()
    conn.close()

def verify_code(email, code):
    """Verify the code for password reset"""
    conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id FROM verification_codes 
        WHERE email = ? AND code = ? AND expires_at > CURRENT_TIMESTAMP AND used = 0
    ''', (email, code))
    
    result = cursor.fetchone()
    
    if result:
        # Mark code as used
        cursor.execute('UPDATE verification_codes SET used = 1 WHERE id = ?', (result[0],))
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

def update_password(email, new_password):
    """Update admin password"""
    conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
    cursor = conn.cursor()
    
    password_hash = generate_password_hash(new_password)
    cursor.execute('UPDATE admins SET password_hash = ? WHERE email = ?', (password_hash, email))
    
    conn.commit()
    conn.close()


def send_verification_email(email, code):
    """Send verification code via email (mock implementation)"""
    print(f"Verification email sent to {email}: Your verification code is {code}")
    print("Note: In production, this would be sent via SMTP server")
    return True


# =================== NEW REQUEST DATABASE FUNCTIONS ===================

def get_token_requests():
    """Get all token requests from request.db - FIXED to handle missing tables"""
    try:
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='token_requests'
        """)
        if not cursor.fetchone():
            conn.close()
            return []
        
        cursor.execute('''
            SELECT id, 
                   COALESCE(user_name, 'Unknown User') as user_name,
                   COALESCE(user_email, '') as user_email, 
                   COALESCE(user_department, 'Unknown') as user_department,
                   COALESCE(tokens_requested, 0) as tokens_requested,
                   COALESCE(priority, 'medium') as priority,
                   COALESCE(reason, '') as reason,
                   COALESCE(status, 'pending') as status,
                   COALESCE(created_at, request_timestamp, datetime('now')) as created_at,
                   COALESCE(current_tokens_used, 0) as current_tokens_used,
                   COALESCE(current_token_limit, 1000) as current_token_limit
            FROM token_requests 
            WHERE status = 'pending'
            ORDER BY created_at DESC
        ''')
        
        requests = []
        for row in cursor.fetchall():
            requests.append({
                'id': row[0],
                'employee_name': row[1],
                'employee_email': row[2],
                'department': row[3],
                'tokens_requested': row[4],
                'priority': row[5],
                'reason': row[6],
                'status': row[7],
                'created_at': row[8],
                'current_usage': row[9],
                'total_limit': row[10]
            })
        
        conn.close()
        return requests
        
    except sqlite3.Error as e:
        print(f"Database error in get_token_requests: {e}")
        return []
    except Exception as e:
        print(f"Error fetching token requests: {e}")
        return []

# =================== USER MANAGEMENT FUNCTIONS ===================

def get_all_users():
    """Get all users from both users and admins tables - FIXED to prevent duplicates"""
    try:
        users = []
        
        # First, get users from the users table (users.db)
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Check if users table exists
        cursor.execute("PRAGMA table_info(users);")
        users_table_exists = cursor.fetchall()
        
        if users_table_exists:
            cursor.execute('''
                SELECT id, name, email, department, status, last_login, created_at
                FROM users
                ORDER BY created_at DESC
            ''')
            
            for row in cursor.fetchall():
                users.append({
                    'id': f"user_{row[0]}",  # Prefix to distinguish from admin IDs
                    'name': row[1],
                    'email': row[2],
                    'department': row[3],
                    'status': row[4],
                    'last_login': row[5],
                    'created_at': row[6],
                    'source': 'users'  # To track which table this came from
                })
        
        conn.close()
        
        # Then, get users from the admins table (admins.db) - FIXED: Use correct database
        conn = sqlite3.connect(ADMIN_DATABASE)  # FIXED: Use ADMIN_DATABASE instead of DATABASE
        cursor = conn.cursor()
        
        # Check if admins table exists
        cursor.execute("PRAGMA table_info(admins);")
        admins_table_exists = cursor.fetchall()
        
        if admins_table_exists:
            cursor.execute('''
                SELECT id, name, email, department, 
                       CASE WHEN is_active = 1 THEN 'active' ELSE 'suspended' END as status,
                       last_login, created_at
                FROM admins
                ORDER BY created_at DESC
            ''')
            
            for row in cursor.fetchall():
                users.append({
                    'id': f"admin_{row[0]}",  # Prefix to distinguish from user IDs
                    'name': row[1],
                    'email': row[2],
                    'department': row[3],
                    'status': row[4],
                    'last_login': row[5],
                    'created_at': row[6],
                    'source': 'admins'  # To track which table this came from
                })
        
        conn.close()
        
        # Remove duplicates based on email (keep the most recent entry)
        seen_emails = set()
        unique_users = []
        
        # Sort by created_at descending to keep the most recent entries
        users.sort(key=lambda x: x['created_at'] or '', reverse=True)
        
        for user in users:
            if user['email'] not in seen_emails:
                seen_emails.add(user['email'])
                unique_users.append(user)
        
        return unique_users
        
    except sqlite3.OperationalError as e:
        print(f"❌ Database error: {e}")
        return []
    except Exception as e:
        print(f"❌ Error fetching users: {e}")
        return []
  
def get_all_admins():
    """Get all admins from ADMIN database"""
    try:
        conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, email, department, 
                   CASE WHEN is_active = 1 THEN 'active' ELSE 'suspended' END as status,
                   last_login, created_at
            FROM admins
            ORDER BY created_at DESC
        ''')
        
        admins = []
        for row in cursor.fetchall():
            admins.append({
                'id': row[0],
                'name': row[1],
                'email': row[2],
                'department': row[3],
                'status': row[4],
                'last_login': row[5],
                'created_at': row[6]
            })
        
        conn.close()
        return admins
    except Exception as e:
        print(f"Error fetching admins: {e}")
        return []

def create_new_user(name, email, department):
    """Add new user to users table - UPDATED to ensure it goes to users table"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Check if email already exists in users table
        cursor.execute('SELECT email FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return {'success': False, 'error': 'Email already exists in users table'}
        
        # Check if email exists in admins table
        admin_conn = sqlite3.connect(ADMIN_DATABASE)
        admin_cursor = admin_conn.cursor()
        admin_cursor.execute('SELECT email FROM admins WHERE email = ?', (email,))
        if admin_cursor.fetchone():
            admin_conn.close()
            conn.close()
            return {'success': False, 'error': 'Email already exists in admin database'}
        admin_conn.close()
        
        # Add to users table
        cursor.execute('''
            INSERT INTO users (name, email, department, status, is_active, created_at)
            VALUES (?, ?, ?, 'active', 1, CURRENT_TIMESTAMP)
        ''', (name, email, department))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {'success': True, 'user_id': user_id}
    except sqlite3.IntegrityError:
        return {'success': False, 'error': 'Email already exists'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_user_by_id(user_id):
    """Get admin user by ID"""
    try:
        conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, email, password_hash, name, department, last_login, is_active
            FROM admins WHERE id = ? AND is_active = 1
        ''', (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'email': user[1],
                'password_hash': user[2],
                'name': user[3],
                'department': user[4],
                'last_login': user[5],
                'is_active': user[6]
            }
        return None
    except Exception as e:
        print(f"Error getting user by ID: {e}")
        return None


def add_admin(name, email, department, password):
    """Add new admin to ADMIN database"""
    try:
        conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO admins (name, email, department, password_hash)
            VALUES (?, ?, ?, ?)
        ''', (name, email, department, password_hash))
        admin_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {'success': True, 'admin_id': admin_id}
    except sqlite3.IntegrityError:
        return {'success': False, 'error': 'Email already exists'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def update_user(user_id, name, email, department):
    """Update user information - FIXED to handle both databases correctly"""
    try:
        # Check if this is a user from users table or admins table
        if user_id.startswith('user_'):
            actual_id = user_id.replace('user_', '')
            conn = sqlite3.connect(DATABASE)  # users.db
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET name = ?, email = ?, department = ?
                WHERE id = ?
            ''', (name, email, department, actual_id))
        elif user_id.startswith('admin_'):
            actual_id = user_id.replace('admin_', '')
            conn = sqlite3.connect(ADMIN_DATABASE)  # admins.db
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE admins SET name = ?, email = ?, department = ?
                WHERE id = ?
            ''', (name, email, department, actual_id))
        else:
            # Fallback: try users table first
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE id = ?', (user_id,))
            if cursor.fetchone():
                cursor.execute('''
                    UPDATE users SET name = ?, email = ?, department = ?
                    WHERE id = ?
                ''', (name, email, department, user_id))
            else:
                conn.close()
                # Try admins table
                conn = sqlite3.connect(ADMIN_DATABASE)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE admins SET name = ?, email = ?, department = ?
                    WHERE id = ?
                ''', (name, email, department, user_id))
        
        conn.commit()
        conn.close()
        return {'success': True}
    except sqlite3.IntegrityError:
        return {'success': False, 'error': 'Email already exists'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

  
def update_admin(admin_id, name, email, department):
    """Update admin information"""
    try:
        conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE admins SET name = ?, email = ?, department = ?
            WHERE id = ?
        ''', (name, email, department, admin_id))
        conn.commit()
        conn.close()
        return {'success': True}
    except sqlite3.IntegrityError:
        return {'success': False, 'error': 'Email already exists'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def toggle_user_status(user_id, status):
    """Toggle user status (active/suspended) - FIXED to handle both databases correctly"""
    try:
        # Check if this is a user from users table or admins table
        if user_id.startswith('user_'):
            actual_id = user_id.replace('user_', '')
            conn = sqlite3.connect(DATABASE)  # users.db
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET status = ? WHERE id = ?', (status, actual_id))
        elif user_id.startswith('admin_'):
            actual_id = user_id.replace('admin_', '')
            conn = sqlite3.connect(ADMIN_DATABASE)  # admins.db
            cursor = conn.cursor()
            is_active = 1 if status == 'active' else 0
            cursor.execute('UPDATE admins SET is_active = ? WHERE id = ?', (is_active, actual_id))
        else:
            # Fallback: try users table first
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE id = ?', (user_id,))
            if cursor.fetchone():
                cursor.execute('UPDATE users SET status = ? WHERE id = ?', (status, user_id))
            else:
                conn.close()
                # Try admins table
                conn = sqlite3.connect(ADMIN_DATABASE)
                cursor = conn.cursor()
                is_active = 1 if status == 'active' else 0
                cursor.execute('UPDATE admins SET is_active = ? WHERE id = ?', (is_active, user_id))
        
        conn.commit()
        conn.close()
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def toggle_admin_status(admin_id, status):
    """Toggle admin status (active/suspended)"""
    try:
        conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
        cursor = conn.cursor()
        is_active = 1 if status == 'active' else 0
        cursor.execute('UPDATE admins SET is_active = ? WHERE id = ?', (is_active, admin_id))
        conn.commit()
        conn.close()
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def delete_user(user_id):
    """Delete user from appropriate database - FIXED to handle both databases correctly"""
    try:
        # Check if this is a user from users table or admins table
        if user_id.startswith('user_'):
            actual_id = user_id.replace('user_', '')
            conn = sqlite3.connect(DATABASE)  # users.db
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE id = ?', (actual_id,))
        elif user_id.startswith('admin_'):
            actual_id = user_id.replace('admin_', '')
            conn = sqlite3.connect(ADMIN_DATABASE)  # admins.db
            cursor = conn.cursor()
            cursor.execute('DELETE FROM admins WHERE id = ?', (actual_id,))
        else:
            # Fallback: try users table first
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE id = ?', (user_id,))
            if cursor.fetchone():
                cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            else:
                conn.close()
                # Try admins table
                conn = sqlite3.connect(ADMIN_DATABASE)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM admins WHERE id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def delete_admin(admin_id):
    """Delete admin from ADMIN database"""
    try:
        conn = sqlite3.connect(ADMIN_DATABASE)  # CHANGED
        cursor = conn.cursor()
        cursor.execute('DELETE FROM admins WHERE id = ?', (admin_id,))
        conn.commit()
        conn.close()
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}
    
def grant_tokens_to_employee(employee_email, tokens_to_grant, admin_id):
    """Grant additional tokens to an employee by updating their token limit and creating accepted request record"""
    try:
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        # First, check if employee exists and get current data
        cursor.execute('''
            SELECT DISTINCT current_token_limit, user_name, user_department
            FROM token_requests 
            WHERE user_email = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (employee_email,))
        
        result = cursor.fetchone()
        
        if result:
            current_limit = result[0] or 1000
            user_name = result[1] or 'Unknown User'
            user_department = result[2] or 'Unknown Dept'
            new_limit = current_limit + tokens_to_grant
            
            # Update all existing records for this employee with new token limit
            cursor.execute('''
                UPDATE token_requests 
                SET current_token_limit = ?
                WHERE user_email = ?
            ''', (new_limit, employee_email))
            
        else:
            # If no existing records, set defaults
            current_limit = 1000
            new_limit = current_limit + tokens_to_grant
            user_name = 'Unknown User'
            user_department = 'Unknown Dept'
        
        # Create a new request record showing this admin grant with "accepted" status
        request_timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO token_requests 
            (user_name, user_email, user_department, tokens_requested, current_tokens_used, 
             current_token_limit, priority, reason, status, request_timestamp, processed_at, processed_by)
            VALUES (?, ?, ?, ?, 0, ?, 'admin', 'Direct token grant by admin', 'accepted', ?, ?, ?)
        ''', (
            user_name,
            employee_email,
            user_department,
            tokens_to_grant,
            new_limit,
            request_timestamp,
            request_timestamp,  # processed_at same as request_timestamp
            admin_id
        ))
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'new_total': new_limit,
            'tokens_granted': tokens_to_grant,
            'status': 'accepted'  # This will show in token requests as accepted
        }
        
    except sqlite3.Error as e:
        print(f"Database error in grant_tokens_to_employee: {e}")
        return {'success': False, 'error': 'Database error occurred'}
    except Exception as e:
        print(f"Error in grant_tokens_to_employee: {e}")
        return {'success': False, 'error': 'An error occurred while granting tokens'}

def get_feedback_requests():
    """Get all feedback requests from request.db - UPDATED with better error handling"""
    try:
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='incorrect_feedback'
        """)
        if not cursor.fetchone():
            conn.close()
            print("⚠️ incorrect_feedback table does not exist")
            return []
        
        cursor.execute('''
            SELECT id,
                   COALESCE(user_name, 'Unknown User') as user_name,
                   COALESCE(user_email, '') as user_email,
                   COALESCE(user_department, 'Unknown') as user_department,
                   COALESCE(original_query, '') as original_query,
                   COALESCE(original_response, '') as original_response,
                   COALESCE(corrected_answer, '') as corrected_answer,
                   COALESCE(status, 'pending') as status,
                   COALESCE(feedback_timestamp, datetime('now')) as feedback_timestamp,
                   COALESCE(reviewed_at, '') as reviewed_at,
                   COALESCE(reviewed_by, '') as reviewed_by
            FROM incorrect_feedback 
            WHERE status IN ('pending', 'accepted', 'rejected', 'on_hold')
            ORDER BY feedback_timestamp DESC
        ''')
        
        requests = []
        for row in cursor.fetchall():
            requests.append({
                'id': row[0],
                'employee_name': row[1],
                'employee_email': row[2],
                'department': row[3],
                'question': row[4],          # original_query
                'db_response': row[5],       # original_response  
                'user_response': row[6],     # corrected_answer
                'status': row[7],
                'created_at': row[8],        # feedback_timestamp
                'reviewed_at': row[9],
                'reviewed_by': row[10]
            })
        
        conn.close()
        print(f"✅ Fetched {len(requests)} feedback requests")
        return requests
        
    except sqlite3.Error as e:
        print(f"❌ Database error in get_feedback_requests: {e}")
        return []
    except Exception as e:
        print(f"❌ Error fetching feedback requests: {e}")
        return []
# Add these updated functions to your Flask app.py
def init_request_db():
    """Initialize request database with required tables"""
    try:
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        # Create token_requests table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL,
                user_email TEXT NOT NULL,
                user_department TEXT NOT NULL,
                tokens_requested INTEGER NOT NULL,
                current_tokens_used INTEGER DEFAULT 0,
                current_token_limit INTEGER DEFAULT 1000,
                priority TEXT DEFAULT 'medium',
                reason TEXT,
                status TEXT DEFAULT 'pending',
                request_timestamp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                processed_by INTEGER
            )
        ''')
        
        # Create incorrect_feedback table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incorrect_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL,
                user_email TEXT NOT NULL,
                user_department TEXT NOT NULL,
                original_query TEXT NOT NULL,
                original_response TEXT NOT NULL,
                corrected_answer TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                feedback_timestamp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                reviewed_by INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Request database initialized successfully")
        
    except sqlite3.Error as e:
        print(f"Error initializing request database: {e}")
    except Exception as e:
        print(f"Unexpected error initializing request database: {e}")
        
# Add these updated functions to your Flask app.py

def get_employee_data():
    """Get employee token usage data - FIXED to handle missing user names"""
    try:
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        # Check if the token_requests table exists and has data
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('token_requests', 'incorrect_feedback')
        """)
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        employees = []
        
        if 'token_requests' in existing_tables:
            # Get employees from token_requests table with LATEST request data
            # Using ROW_NUMBER() to get the most recent record per employee
            cursor.execute('''
                WITH LatestRequests AS (
                    SELECT 
                        user_name,
                        COALESCE(user_email, '') as email,
                        COALESCE(user_department, 'Unknown') as department,
                        COALESCE(current_tokens_used, 0) as current_usage,
                        COALESCE(current_token_limit, 1000) as total_limit,
                        COALESCE(created_at, request_timestamp, datetime('now')) as created_at,
                        ROW_NUMBER() OVER (PARTITION BY user_email ORDER BY 
                            COALESCE(created_at, request_timestamp, datetime('now')) DESC) as rn
                    FROM token_requests
                    WHERE user_email IS NOT NULL AND user_email != ''
                )
                SELECT user_name, email, department, current_usage, total_limit
                FROM LatestRequests 
                WHERE rn = 1
            ''')
            
            token_employees = cursor.fetchall()
            for row in token_employees:
                email = row[1]
                current_usage = row[3]
                total_limit = row[4]
                
                # Handle user name - extract from email if NULL/empty
                user_name = row[0]
                if not user_name or user_name.strip() == '' or user_name.lower() == 'unknown user':
                    # Extract name from email (part before @)
                    email_name = email.split('@')[0] if email else 'Unknown User'
                    # Capitalize and replace dots/underscores with spaces
                    display_name = email_name.replace('.', ' ').replace('_', ' ').title()
                else:
                    display_name = user_name.strip()
                
                # Get pending requests count for this employee
                cursor.execute('''
                    SELECT COUNT(*) FROM token_requests 
                    WHERE user_email = ? AND status = 'pending'
                ''', (email,))
                pending_count = cursor.fetchone()[0]
                
                # Get on-hold requests count for this employee
                cursor.execute('''
                    SELECT COUNT(*) FROM token_requests 
                    WHERE user_email = ? AND status = 'on_hold'
                ''', (email,))
                hold_count = cursor.fetchone()[0]
                
                # Get the latest token limit from the most recent request
                cursor.execute('''
                    SELECT current_token_limit, tokens_requested, status
                    FROM token_requests 
                    WHERE user_email = ? 
                    ORDER BY COALESCE(created_at, request_timestamp, datetime('now')) DESC 
                    LIMIT 1
                ''', (email,))
                latest_request = cursor.fetchone()
                
                if latest_request:
                    latest_limit = latest_request[0] or 1000
                    latest_requested = latest_request[1] or 0
                    latest_status = latest_request[2] or 'pending'
                else:
                    latest_limit = total_limit
                    latest_requested = 0
                    latest_status = 'active'
                
                # Determine status based on requests and limits
                if pending_count > 0:
                    final_status = 'pending'
                elif hold_count > 0:
                    final_status = 'hold'
                elif current_usage >= latest_limit:
                    final_status = 'limit-reached'
                elif current_usage >= latest_limit * 0.8:
                    final_status = 'near-limit'
                else:
                    final_status = 'active'
                
                employees.append({
                    'name': display_name,  # Use the cleaned display name
                    'email': email,
                    'department': row[2],
                    'current_usage': current_usage,
                    'total_limit': latest_limit,
                    'status': final_status,
                    'pending_requests': pending_count,
                    'on_hold_requests': hold_count,
                    'latest_requested_tokens': latest_requested,
                    'latest_request_status': latest_status
                })
        
        if 'incorrect_feedback' in existing_tables:
            # Get employees from feedback table who aren't already in the list
            existing_emails = [emp['email'] for emp in employees]
            
            if existing_emails:
                placeholders = ','.join(['?' for _ in existing_emails])
                cursor.execute(f'''
                    SELECT DISTINCT 
                        user_name,
                        COALESCE(user_email, '') as email,
                        COALESCE(user_department, 'Unknown') as department
                    FROM incorrect_feedback
                    WHERE user_email IS NOT NULL AND user_email != ''
                    AND user_email NOT IN ({placeholders})
                ''', existing_emails)
            else:
                cursor.execute('''
                    SELECT DISTINCT 
                        user_name,
                        COALESCE(user_email, '') as email,
                        COALESCE(user_department, 'Unknown') as department
                    FROM incorrect_feedback
                    WHERE user_email IS NOT NULL AND user_email != ''
                ''')
            
            feedback_employees = cursor.fetchall()
            for row in feedback_employees:
                user_name = row[0]
                email = row[1]
                
                # Handle user name for feedback users too
                if not user_name or user_name.strip() == '' or user_name.lower() == 'unknown user':
                    email_name = email.split('@')[0] if email else 'Unknown User'
                    display_name = email_name.replace('.', ' ').replace('_', ' ').title()
                else:
                    display_name = user_name.strip()
                
                employees.append({
                    'name': display_name,
                    'email': email,
                    'department': row[2],
                    'current_usage': 0,
                    'total_limit': 1000,
                    'status': 'active',
                    'pending_requests': 0,
                    'on_hold_requests': 0,
                    'latest_requested_tokens': 0,
                    'latest_request_status': 'none'
                })
        
        # If no employees found, add some sample data to prevent errors
        if not employees:
            employees = [{
                'name': 'No employees found',
                'email': 'no-data@shrirampistons.com',
                'department': 'N/A',
                'current_usage': 0,
                'total_limit': 1000,
                'status': 'active',
                'pending_requests': 0,
                'on_hold_requests': 0,
                'latest_requested_tokens': 0,
                'latest_request_status': 'none'
            }]
        
        conn.close()
        return employees
        
    except sqlite3.Error as e:
        print(f"Database error in get_employee_data: {e}")
        # Return empty data structure to prevent JavaScript errors
        return [{
            'name': 'Database Error',
            'email': 'error@shrirampistons.com',
            'department': 'System',
            'current_usage': 0,
            'total_limit': 1000,
            'status': 'error',
            'pending_requests': 0,
            'on_hold_requests': 0,
            'latest_requested_tokens': 0,
            'latest_request_status': 'error'
        }]
    except Exception as e:
        print(f"Error fetching employee data: {e}")
        return [{
            'name': 'System Error',
            'email': 'system@shrirampistons.com', 
            'department': 'System',
            'current_usage': 0,
            'total_limit': 1000,
            'status': 'error',
            'pending_requests': 0,
            'on_hold_requests': 0,
            'latest_requested_tokens': 0,
            'latest_request_status': 'error'
        }]

def update_request_status(request_type, request_id, status, admin_id=None, grant_tokens=False):
    """Update request status in database - ENHANCED to properly update token limits"""
    try:
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        if request_type == 'token':
            # Handle token requests
            if status == 'accepted' and grant_tokens:
                # Get request details first
                cursor.execute('''
                    SELECT user_email, tokens_requested, current_token_limit 
                    FROM token_requests WHERE id = ?
                ''', (request_id,))
                request_data = cursor.fetchone()
                
                if request_data:
                    user_email, tokens_requested, current_limit = request_data
                    new_limit = (current_limit or 1000) + tokens_requested
                    
                    # Update the CURRENT request record with new status and processed info
                    cursor.execute('''
                        UPDATE token_requests 
                        SET status = ?, processed_at = CURRENT_TIMESTAMP, processed_by = ?,
                            current_token_limit = ?
                        WHERE id = ?
                    ''', (status, admin_id, new_limit, request_id))
                    
                    # Create a NEW record to reflect the updated token allocation
                    request_timestamp = datetime.now().isoformat()
                    cursor.execute('''
                        INSERT INTO token_requests 
                        (user_name, user_email, user_department, tokens_requested, current_tokens_used, 
                         current_token_limit, priority, reason, status, request_timestamp, processed_at, processed_by)
                        SELECT user_name, user_email, user_department, 0, current_tokens_used,
                               ?, 'system', 'Token limit updated after admin approval', 'granted', ?, ?, ?
                        FROM token_requests WHERE id = ?
                    ''', (
                        new_limit,           # new current_token_limit
                        request_timestamp,   # request_timestamp
                        request_timestamp,   # processed_at (same as request_timestamp)
                        admin_id,           # processed_by
                        request_id          # from the original request
                    ))
                    
                    print(f"✅ Granted {tokens_requested} tokens to {user_email}. New limit: {new_limit}")
                else:
                    print("❌ Request not found for token granting")
            else:
                # Just update the request status without granting tokens
                cursor.execute('''
                    UPDATE token_requests 
                    SET status = ?, processed_at = CURRENT_TIMESTAMP, processed_by = ?
                    WHERE id = ?
                ''', (status, admin_id, request_id))
            
        elif request_type == 'feedback':
            # Handle feedback requests - FIXED to use correct table
            cursor.execute('''
                UPDATE incorrect_feedback 
                SET status = ?, reviewed_at = CURRENT_TIMESTAMP, reviewed_by = ?
                WHERE id = ?
            ''', (status, admin_id, request_id))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        if rows_affected > 0:
            print(f"✅ Updated {request_type} request {request_id} to status: {status}")
            return True
        else:
            print(f"⚠️ No rows affected when updating {request_type} request {request_id}")
            return False
        
    except sqlite3.Error as e:
        print(f"❌ Database error updating request status: {e}")
        return False
    except Exception as e:
        print(f"❌ Error updating request status: {e}")
        return False

# =================== ADD THESE NEW ROUTES ===================
@app.route('/api/sync-excel', methods=['POST'])
def api_sync_excel():
    """Manually sync admins from Excel file"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    success = sync_admins_from_excel()
    if success:
        return jsonify({'success': True, 'message': 'Admins synced from Excel successfully'})
    else:
        return jsonify({'success': False, 'error': 'Failed to sync from Excel'}), 500

@app.route('/api/export-excel', methods=['POST'])
def api_export_excel():
    """Export current admins to Excel file"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    success = export_admins_to_excel()
    if success:
        return jsonify({'success': True, 'message': 'Admins exported to Excel successfully'})
    else:
        return jsonify({'success': False, 'error': 'Failed to export to Excel'}), 500

@app.route('/api/check-excel-changes')
def api_check_excel_changes():
    """Check if Excel file has been modified"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        current_hash = get_file_hash(ADMINS_EXCEL_FILE)
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT last_hash FROM file_sync WHERE filename = ?', (ADMINS_EXCEL_FILE,))
        result = cursor.fetchone()
        conn.close()
        
        stored_hash = result[0] if result else None
        
        return jsonify({
            'success': True,
            'file_exists': current_hash is not None,
            'has_changes': current_hash != stored_hash,
            'needs_sync': current_hash != stored_hash
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/admin-token-requests')
def api_admin_token_requests():
    """API endpoint to get token requests"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    requests = get_token_requests()
    return jsonify({
        'success': True,
        'requests': requests,
        'count': len(requests)
    })
    
# =================== USER MANAGEMENT API ROUTES ===================

@app.route('/api/users', methods=['GET'])
def api_get_users():
    if 'user_id' not in session or not session.get('user_id'):
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        users = get_all_users()
        return jsonify({
            'success': True, 
            'users': users,
            'count': len(users)
        })
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/debug/test-users')
def debug_test_users():
    """Test users API"""
    try:
        users = get_all_users()
        return jsonify({
            'success': True,
            'users': users,
            'count': len(users)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admins', methods=['GET'])
def api_get_admins():
    """Get all admins"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    admins = get_all_admins()
    return jsonify({'success': True, 'admins': admins})


@app.route('/api/add-user', methods=['POST'])
def api_add_user():
    """Add new user - FIXED API route"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').lower().strip()
        department = data.get('department', '').strip()
        
        # Validate input
        if not all([name, email, department]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        # Validate email format
        if not validate_shriram_email(email):
            return jsonify({'success': False, 'error': 'Please use a valid @shrirampistons.com email address'}), 400
        
        # Create the user
        result = create_new_user(name, email, department)
        
        if result['success']:
            return jsonify({
                'success': True, 
                'message': 'User added successfully',
                'user_id': result['user_id']
            })
        else:
            return jsonify({'success': False, 'error': result['error']}), 400
            
    except Exception as e:
        print(f"Error in api_add_user: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.route('/api/add-admin', methods=['POST'])
def api_add_admin():
    """Add new admin"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    name = data.get('name', '').strip()
    email = data.get('email', '').lower().strip()
    department = data.get('department', '').strip()
    password = data.get('password', '').strip()
    
    if not all([name, email, department, password]):
        return jsonify({'success': False, 'error': 'All fields are required'}), 400
    
    if not validate_shriram_email(email):
        return jsonify({'success': False, 'error': 'Invalid email domain'}), 400
    
    if len(password) < 6:
        return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
    
    result = add_admin(name, email, department, password)
    if result['success']:
        return jsonify({'success': True, 'message': 'Admin added successfully'})
    else:
        return jsonify({'success': False, 'error': result['error']}), 400

@app.route('/api/edit-user', methods=['POST'])
def api_edit_user():
    """Edit user"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    user_id = data.get('id')
    name = data.get('name', '').strip()
    email = data.get('email', '').lower().strip()
    department = data.get('department', '').strip()
    
    if not all([user_id, name, email, department]):
        return jsonify({'success': False, 'error': 'All fields are required'}), 400
    
    result = update_user(user_id, name, email, department)
    if result['success']:
        return jsonify({'success': True, 'message': 'User updated successfully'})
    else:
        return jsonify({'success': False, 'error': result['error']}), 400

@app.route('/api/edit-admin', methods=['POST'])
def api_edit_admin():
    """Edit admin"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    admin_id = data.get('id')
    name = data.get('name', '').strip()
    email = data.get('email', '').lower().strip()
    department = data.get('department', '').strip()
    
    if not all([admin_id, name, email, department]):
        return jsonify({'success': False, 'error': 'All fields are required'}), 400
    
    result = update_admin(admin_id, name, email, department)
    if result['success']:
        return jsonify({'success': True, 'message': 'Admin updated successfully'})
    else:
        return jsonify({'success': False, 'error': result['error']}), 400

@app.route('/api/suspend-user', methods=['POST'])
def api_suspend_user():
    """Suspend/activate user"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    user_id = data.get('id')
    status = data.get('status')  # 'active' or 'suspended'
    
    if not user_id or status not in ['active', 'suspended']:
        return jsonify({'success': False, 'error': 'Invalid parameters'}), 400
    
    result = toggle_user_status(user_id, status)
    if result['success']:
        action = 'activated' if status == 'active' else 'suspended'
        return jsonify({'success': True, 'message': f'User {action} successfully'})
    else:
        return jsonify({'success': False, 'error': result['error']}), 400

@app.route('/api/suspend-admin', methods=['POST'])
def api_suspend_admin():
    """Suspend/activate admin"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    admin_id = data.get('id')
    status = data.get('status')  # 'active' or 'suspended'
    
    if not admin_id or status not in ['active', 'suspended']:
        return jsonify({'success': False, 'error': 'Invalid parameters'}), 400
    
    result = toggle_admin_status(admin_id, status)
    if result['success']:
        action = 'activated' if status == 'active' else 'suspended'
        return jsonify({'success': True, 'message': f'Admin {action} successfully'})
    else:
        return jsonify({'success': False, 'error': result['error']}), 400

@app.route('/api/remove-user', methods=['DELETE'])
def api_remove_user():
    """Remove user - UPDATED to prevent admin deletion from user interface"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    user_id = data.get('id')
    
    if not user_id:
        return jsonify({'success': False, 'error': 'User ID required'}), 400
    
    # Prevent deletion of admins from user interface
    if user_id.startswith('admin_'):
        return jsonify({
            'success': False, 
            'error': 'Cannot delete admin users from this interface. Use Admin Database tab.'
        }), 400
    
    # Prevent self-deletion if current user somehow appears in user list
    current_admin_id = f"admin_{session.get('user_id')}"
    if user_id == current_admin_id:
        return jsonify({'success': False, 'error': 'Cannot delete your own account'}), 400
    
    result = delete_user(user_id)
    if result['success']:
        return jsonify({'success': True, 'message': 'User removed successfully'})
    else:
        return jsonify({'success': False, 'error': result['error']}), 400
    
@app.route('/api/remove-admin', methods=['DELETE'])
def api_remove_admin():
    """Remove admin"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    admin_id = data.get('id')
    
    if not admin_id:
        return jsonify({'success': False, 'error': 'Admin ID required'}), 400
    
    # Prevent self-deletion
    if admin_id == session.get('user_id'):
        return jsonify({'success': False, 'error': 'Cannot delete your own account'}), 400
    
    result = delete_admin(admin_id)
    if result['success']:
        return jsonify({'success': True, 'message': 'Admin removed successfully'})
    else:
        return jsonify({'success': False, 'error': result['error']}), 400
    

@app.route('/api/admin-feedback-requests')
def api_admin_feedback_requests():
    """API endpoint to get feedback requests"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    requests = get_feedback_requests()
    return jsonify({
        'success': True,
        'requests': requests,
        'count': len(requests)
    })

@app.route('/api/admin-employee-data')
def api_admin_employee_data():
    """API endpoint to get employee data"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    employees = get_employee_data()
    return jsonify({
        'success': True,
        'employees': employees,
        'count': len(employees)
    })
    
@app.route('/api/admin-grant-tokens', methods=['POST'])
def api_admin_grant_tokens():
    """API endpoint to grant tokens directly to employees"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        employee_email = data.get('employee_email')
        tokens_to_grant = data.get('tokens_to_grant')
        current_total = data.get('current_total', 0)
        admin_id = session.get('user_id')
        
        # Validation
        if not employee_email or not tokens_to_grant:
            return jsonify({
                'success': False,
                'error': 'Employee email and tokens to grant are required'
            }), 400
        
        # Validate email domain
        if not validate_shriram_email(employee_email):
            return jsonify({
                'success': False,
                'error': 'Please use a valid @shrirampistons.com email address'
            }), 400
        
        # Validate token amount
        try:
            tokens_to_grant = int(tokens_to_grant)
            if tokens_to_grant <= 0 or tokens_to_grant > 10000:
                return jsonify({
                    'success': False,
                    'error': 'Tokens to grant must be between 1 and 10000'
                }), 400
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid token amount provided'
            }), 400
        
        # Grant tokens
        result = grant_tokens_to_employee(employee_email, tokens_to_grant, admin_id)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': f'Successfully granted {tokens_to_grant} tokens to {employee_email}',
                'new_total': result['new_total'],
                'tokens_granted': result['tokens_granted']
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to grant tokens')
            }), 500
            
    except Exception as e:
        print(f"Error in admin grant tokens: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while granting tokens'
        }), 500

@app.route('/api/admin-process-request', methods=['POST'])
def api_admin_process_request():
    """API endpoint to process (accept/reject) requests - UPDATED to handle token granting"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        request_type = data.get('type')  # 'token' or 'feedback'
        request_id = data.get('id')
        action = data.get('action')  # 'accept', 'reject', 'hold'
        grant_tokens = data.get('grant_tokens', False)  # NEW: whether to grant tokens when accepting
        admin_id = session.get('user_id')
        
        if not all([request_type, request_id, action]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400
        
        # Map action to status
        status_map = {
            'accept': 'accepted',
            'reject': 'rejected',
            'hold': 'on_hold'
        }
        
        status = status_map.get(action)
        if not status:
            return jsonify({
                'success': False,
                'error': 'Invalid action'
            }), 400
        
        # Update request status with token granting option
        success = update_request_status(request_type, request_id, status, admin_id, grant_tokens)
        
        if success:
            message = f'Request {action}ed successfully'
            if action == 'accept' and grant_tokens and request_type == 'token':
                message += ' and tokens granted to user'
                
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update request status'
            }), 500
            
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing the request'
        }), 500

@app.route('/api/submit-token-request', methods=['POST'])
def submit_token_request():
    """Handle token request submissions from chatbot"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['employee_name', 'employee_email', 'department', 'tokens_requested', 'reason']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate email domain
        if not validate_shriram_email(data['employee_email']):
            return jsonify({
                'success': False,
                'error': 'Please use a valid @shrirampistons.com email address'
            }), 400
        
        # Validate tokens requested
        tokens_requested = int(data.get('tokens_requested', 0))
        if tokens_requested <= 0 or tokens_requested > 5000:
            return jsonify({
                'success': False,
                'error': 'Tokens requested must be between 1 and 5000'
            }), 400
        
        # Get current usage and limits
        current_usage = int(data.get('current_usage', 0))
        total_limit = int(data.get('total_limit', 1000))
        priority = data.get('priority', 'medium').lower()
        
        # Validate priority
        if priority not in ['high', 'medium', 'low']:
            priority = 'medium'
        
        # Insert into database
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO token_requests 
            (employee_name, employee_email, department, tokens_requested, current_usage, 
             total_limit, priority, reason, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        ''', (
            data['employee_name'],
            data['employee_email'].lower().strip(),
            data['department'],
            tokens_requested,
            current_usage,
            total_limit,
            priority,
            data['reason']
        ))
        
        request_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Token request submitted successfully',
            'request_id': request_id,
            'status': 'pending'
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid token amount provided'
        }), 400
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({
            'success': False,
            'error': 'Database error occurred'
        }), 500
    except Exception as e:
        print(f"Error submitting token request: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while submitting your request'
        }), 500
    
@app.route('/api/chatbot-submit-token-request', methods=['POST'])
def chatbot_submit_token_request():
    """Handle token request submissions from chatbot - UPDATED for standardized schema"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['user_name', 'user_email', 'user_department', 'tokens_requested', 'reason']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate email domain
        if not validate_shriram_email(data['user_email']):
            return jsonify({
                'success': False,
                'error': 'Please use a valid @shrirampistons.com email address'
            }), 400
        
        # Validate tokens requested
        tokens_requested = int(data.get('tokens_requested', 0))
        if tokens_requested <= 0 or tokens_requested > 5000:
            return jsonify({
                'success': False,
                'error': 'Tokens requested must be between 1 and 5000'
            }), 400
        
        # Get current usage and limits
        current_tokens_used = int(data.get('current_tokens_used', 0))
        current_token_limit = int(data.get('current_token_limit', 1000))
        priority = data.get('priority', 'medium').lower()
        
        # Validate priority
        if priority not in ['high', 'medium', 'low']:
            priority = 'medium'
        
        # Insert into database with STANDARDIZED column names
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        request_timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO token_requests 
            (user_name, user_email, user_department, tokens_requested, current_tokens_used, 
             current_token_limit, priority, reason, status, request_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        ''', (
            data['user_name'],                              # UPDATED: standardized column name
            data['user_email'].lower().strip(),             # UPDATED: standardized column name
            data['user_department'],                        # UPDATED: standardized column name
            tokens_requested,
            current_tokens_used,                            # UPDATED: standardized column name
            current_token_limit,                            # UPDATED: standardized column name
            priority,
            data['reason'],
            request_timestamp                               # UPDATED: standardized column name
        ))
        
        request_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Token request submitted successfully',
            'request_id': request_id,
            'status': 'pending'
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid token amount provided'
        }), 400
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({
            'success': False,
            'error': 'Database error occurred'
        }), 500
    except Exception as e:
        print(f"Error submitting token request: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while submitting your request'
        }), 500


@app.route('/api/chatbot-submit-feedback-request', methods=['POST']) 
def chatbot_submit_feedback_request():
    """Handle feedback request submissions from chatbot - UPDATED for standardized schema"""
    try:
        data = request.get_json()
        
        # Validate required fields - UPDATED field names
        required_fields = ['user_name', 'user_email', 'user_department', 'original_query', 'original_response', 'corrected_answer']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate email domain
        if not validate_shriram_email(data['user_email']):
            return jsonify({
                'success': False,
                'error': 'Please use a valid @shrirampistons.com email address'
            }), 400
        
        # Insert into database with STANDARDIZED column names
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        feedback_timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO incorrect_feedback 
            (user_name, user_email, user_department, original_query, original_response, corrected_answer, status, feedback_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
        ''', (
            data['user_name'],                              # UPDATED: standardized column name
            data['user_email'].lower().strip(),             # UPDATED: standardized column name
            data['user_department'],                        # UPDATED: standardized column name
            data['original_query'],                         # UPDATED: standardized column name
            data['original_response'],                      # UPDATED: standardized column name
            data['corrected_answer'],                       # UPDATED: standardized column name
            feedback_timestamp                              # UPDATED: standardized column name
        ))
        
        request_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully',
            'request_id': request_id,
            'status': 'pending'
        })
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({
            'success': False,
            'error': 'Database error occurred'
        }), 500
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while submitting your feedback'
        }), 500
        
@app.route('/api/chatbot-get-employee-status', methods=['GET'])
def chatbot_get_employee_status():
    """Get employee's current token status"""
    employee_email = request.args.get('email')
    
    if not employee_email:
        return jsonify({
            'success': False,
            'error': 'Employee email is required'
        }), 400

@app.route('/api/submit-feedback-request', methods=['POST'])
def submit_feedback_request():
    """Handle feedback request submissions - FIXED to use correct table and schema"""
    try:
        data = request.get_json()
        
        # Validate required fields - UPDATED to match new schema
        required_fields = ['user_name', 'user_email', 'user_department', 'original_query', 'original_response', 'corrected_answer']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate email domain
        if not validate_shriram_email(data['user_email']):
            return jsonify({
                'success': False,
                'error': 'Please use a valid @shrirampistons.com email address'
            }), 400
        
        # Insert into database using CORRECT table and column names
        conn = sqlite3.connect(REQUEST_DATABASE)
        cursor = conn.cursor()
        
        feedback_timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO incorrect_feedback 
            (user_name, user_email, user_department, original_query, original_response, corrected_answer, status, feedback_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
        ''', (
            data['user_name'],
            data['user_email'].lower().strip(),
            data['user_department'],
            data['original_query'],
            data['original_response'],
            data['corrected_answer'],
            feedback_timestamp
        ))
        
        request_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully',
            'request_id': request_id,
            'status': 'pending'
        })
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({
            'success': False,
            'error': 'Database error occurred'
        }), 500
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while submitting your feedback'
        }), 500

@app.route('/api/get-employee-status', methods=['GET'])
def get_employee_status():
    """Get employee's current token status"""
    employee_email = request.args.get('email')
    
    if not employee_email:
        return jsonify({
            'success': False,
            'error': 'Employee email is required'
        }), 400

# =================== ORIGINAL ROUTES ===================

@app.route('/')
def index():
    """Serve the login page"""
    # if 'user_id' in session:
    #     return redirect(url_for('admin_portal'))
    try:
        with open('templates/admin_login.html', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>File Not Found</title>
        </head>
        <body>
            <h1>Error: Login page not found</h1>
            <p>Please make sure 'templates/admin_login.html' file is in the same directory as this Flask app.</p>
        </body>
        </html>
        ''', 404

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Authenticate user login"""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({
                'success': False,
                'error': 'Email and password are required'
            }), 400
        
        if not validate_shriram_email(email):
            return jsonify({
                'success': False,
                'error': 'Please use a valid @shrirampistons.com email address'
            }), 400
        
        user = get_user_by_email(email)
        if not user:
            return jsonify({
                'success': False,
                'error': 'Invalid email or password'
            }), 401
        
        if not check_password_hash(user['password_hash'], password):
            return jsonify({
                'success': False,
                'error': 'Invalid email or password'
            }), 401
        
        update_last_login(user['id'])
        
        session.permanent = True
        session['user_id'] = user['id']
        session['user_email'] = user['email']
        session['user_name'] = user['name']
        session['user_department'] = user['department']
        
        return jsonify({
            'success': True,
            'message': f'Welcome, {user["name"]}!',
            'redirect': '/admin-portal',
            'user': {
                'email': user['email'],
                'name': user['name'],
                'department': user['department']
            }
        })
        
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during authentication'
        }), 500

@app.route('/admin-portal')
def admin_portal():
    """Admin portal page - requires authentication"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    try:
        with open('templates/management_portal.html', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        # Return a basic admin portal if the template is missing
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Admin Portal - Shriram Pistons</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
                       padding: 2rem; background: #f8fafc; }}
                .container {{ background: white; padding: 2rem; border-radius: 1rem; 
                             box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
                .btn {{ background: #3b82f6; color: white; padding: 0.75rem 1.5rem; 
                       border: none; border-radius: 0.5rem; text-decoration: none; 
                       display: inline-block; margin: 0.5rem; cursor: pointer; }}
                .btn:hover {{ background: #1d4ed8; }}
                .error-notice {{ background: #fef3c7; border: 1px solid #f59e0b; 
                                color: #92400e; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error-notice">
                    ⚠️ <strong>Template Missing:</strong> The management portal template is not found. 
                    Using basic fallback interface.
                </div>
                
                <h1>🛡️ Admin Portal</h1>
                <p>Welcome, <strong>{session.get('user_name')}</strong> ({session.get('user_email')})</p>
                
                <div style="margin: 2rem 0;">
                    <h3>Available Actions:</h3>
                    <button class="btn" onclick="loadData('employees')">👥 View Employees</button>
                    <button class="btn" onclick="loadData('requests')">📋 View Token Requests</button>
                    <button class="btn" onclick="loadData('feedback')">💬 View Feedback</button>
                    <a href="/logout" class="btn" style="background: #dc2626;">🚪 Logout</a>
                </div>
                
                <div id="data-container" style="margin-top: 2rem;">
                    <p>Click a button above to load data.</p>
                </div>
            </div>
            
            <script>
                let employeeData = [];
                let tokenRequests = [];
                let feedbackRequests = [];
                
                async function loadData(type) {{
                    const container = document.getElementById('data-container');
                    container.innerHTML = '<p>Loading...</p>';
                    
                    try {{
                        let response;
                        switch(type) {{
                            case 'employees':
                                response = await fetch('/api/admin-employee-data');
                                const empData = await response.json();
                                employeeData = empData.employees || [];
                                container.innerHTML = `
                                    <h3>👥 Employee Data (${{employeeData.length}} employees)</h3>
                                    <div>${{employeeData.map(emp => `
                                        <div style="border: 1px solid #e5e7eb; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem;">
                                            <strong>${{emp.name}}</strong> (${{emp.email}})<br>
                                            Department: ${{emp.department}}<br>
                                            Usage: ${{emp.current_usage}}/${{emp.total_limit}} tokens<br>
                                            Status: ${{emp.status}}
                                        </div>
                                    `).join('')}}</div>
                                `;
                                break;
                                
                            case 'requests':
                                response = await fetch('/api/admin-token-requests');
                                const reqData = await response.json();
                                tokenRequests = reqData.requests || [];
                                container.innerHTML = `
                                    <h3>📋 Token Requests (${{tokenRequests.length}} pending)</h3>
                                    <div>${{tokenRequests.map(req => `
                                        <div style="border: 1px solid #e5e7eb; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem;">
                                            <strong>${{req.employee_name}}</strong> (${{req.employee_email}})<br>
                                            Requested: ${{req.tokens_requested}} tokens<br>
                                            Reason: ${{req.reason}}<br>
                                            Priority: ${{req.priority}}
                                        </div>
                                    `).join('')}}</div>
                                `;
                                break;
                                
                            case 'feedback':
                                response = await fetch('/api/admin-feedback-requests');
                                const fbData = await response.json();
                                feedbackRequests = fbData.requests || [];
                                container.innerHTML = `
                                    <h3>💬 Feedback Requests (${{feedbackRequests.length}} items)</h3>
                                    <div>${{feedbackRequests.map(fb => `
                                        <div style="border: 1px solid #e5e7eb; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem;">
                                            <strong>${{fb.employee_name}}</strong> (${{fb.employee_email}})<br>
                                            Question: ${{fb.question}}<br>
                                            Status: ${{fb.status}}
                                        </div>
                                    `).join('')}}</div>
                                `;
                                break;
                        }}
                    }} catch (error) {{
                        container.innerHTML = `<p style="color: #dc2626;">Error loading data: ${{error.message}}</p>`;
                        console.error('Error:', error);
                    }}
                }}
                
                // Load employee data by default
                document.addEventListener('DOMContentLoaded', function() {{
                    loadData('employees');
                }});
            </script>
        </body>
        </html>
        '''

        
@app.route('/chatbot')
def chatbot():
    """Chatbot page - requires authentication"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot - Shriram Pistons</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 1rem;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                padding: 2rem;
            }}
            .header {{
                text-align: center;
                margin-bottom: 2rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #f3f4f6;
            }}
            .user-info {{
                background: #f9fafb;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 2rem;
            }}
            .btn {{
                background: #000;
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 0.5rem;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                margin: 0.5rem;
            }}
            .btn:hover {{
                background: #374151;
            }}
            .btn.admin {{
                background: #3b82f6;
            }}
            .btn.admin:hover {{
                background: #1d4ed8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Shriram Pistons Chatbot</h1>
            </div>
            
            <div class="user-info">
                <h3>User Information</h3>
                <p><strong>Name:</strong> {session.get('user_name')}</p>
                <p><strong>Email:</strong> {session.get('user_email')}</p>
                <p><strong>Department:</strong> {session.get('user_department')}</p>
            </div>
            
            <div style="text-align: center;">
                <h2>Chatbot Interface</h2>
                <p>This is where your chatbot interface would be implemented.</p>
                <br>
                <a href="/admin-portal" class="btn admin">🛡️ Admin Portal</a>
                <a href="/logout" class="btn">🚪 Logout</a>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/send-verification-code', methods=['POST'])
def send_verification_code():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        
        if not email or not validate_shriram_email(email):
            return jsonify({
                'success': False,
                'error': 'Please provide a valid @shrirampistons.com email address'
            }), 400
        
        user = get_user_by_email(email)
        if not user:
            return jsonify({
                'success': False,
                'error': 'Email address not found in our records'
            }), 404
        
        code = str(secrets.randbelow(900000) + 100000)
        store_verification_code(email, code)
        
        if send_verification_email(email, code):
            return jsonify({
                'success': True,
                'message': 'Verification code sent to your email',
                'code': code
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to send verification email'
            }), 500
            
    except Exception as e:
        print(f"Error sending verification code: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while sending verification code'
        }), 500

@app.route('/verify-reset-code', methods=['POST'])
def verify_reset_code():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        code = data.get('code', '').strip()
        
        if not email or not code:
            return jsonify({
                'success': False,
                'error': 'Email and verification code are required'
            }), 400
        
        if verify_code(email, code):
            session['reset_email'] = email
            return jsonify({
                'success': True,
                'message': 'Code verified successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid or expired verification code'
            }), 400
            
    except Exception as e:
        print(f"Error verifying code: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while verifying code'
        }), 500

# Add these route handlers to your Flask app.py for handling chatbot POST requests

@app.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        data = request.get_json()
        new_password = data.get('new_password', '')
        confirm_password = data.get('confirm_password', '')
        
        email = session.get('reset_email')
        if not email:
            return jsonify({
                'success': False,
                'error': 'Session expired. Please start the password reset process again.'
            }), 400
        
        if not new_password or len(new_password) < 6:
            return jsonify({
                'success': False,
                'error': 'Password must be at least 6 characters long'
            }), 400
        
        if new_password != confirm_password:
            return jsonify({
                'success': False,
                'error': 'Passwords do not match'
            }), 400
        
        update_password(email, new_password)
        session.pop('reset_email', None)
        
        return jsonify({
            'success': True,
            'message': 'Password reset successfully. You can now login with your new password.'
        })
        
    except Exception as e:
        print(f"Error resetting password: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while resetting password'
        }), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/user-status')
def user_status():
    if 'user_id' in session:
        return jsonify({
            'logged_in': True,
            'user': {
                'email': session.get('user_email'),
                'name': session.get('user_name'),
                'department': session.get('user_department')
            }
        })
    else:
        return jsonify({'logged_in': False})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Initializing databases...")
    init_db()          # Creates users.db tables and syncs from Excel
    init_request_db()  # Creates request.db tables
    print("✅ Database initialization complete")
    app.run(debug=True)