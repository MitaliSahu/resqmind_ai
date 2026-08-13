import sqlite3
from datetime import datetime

DB_PATH = "data/hospital.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create Patients Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            symptoms TEXT,
            vitals TEXT,
            medical_history TEXT,
            priority TEXT,
            severity_score REAL,
            arrival_time TEXT
        )
    ''')
    
    # Create Resources Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resource_type TEXT,
            total INTEGER,
            available INTEGER
        )
    ''')
    
    # Insert Dummy Resources if Empty
    cursor.execute("SELECT COUNT(*) FROM resources")
    if cursor.fetchone()[0] == 0:
        resources = [
            ("ICU Beds", 20, 5),
            ("General Ward Beds", 100, 45),
            ("Ventilators", 15, 3),
            ("Available Doctors", 10, 4)
        ]
        cursor.executemany("INSERT INTO resources (resource_type, total, available) VALUES (?, ?, ?)", resources)
        
    conn.commit()
    conn.close()
    
def add_patient(name, age, symptoms, vitals, medical_history, priority, severity_score):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    arrival_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO patients (name, age, symptoms, vitals, medical_history, priority, severity_score, arrival_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, age, symptoms, vitals, medical_history, priority, severity_score, arrival_time))
    
    conn.commit()
    conn.close()

def get_queue():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Patients are sorted by severity to handle priority queuing automatically
    cursor.execute("SELECT * FROM patients ORDER BY severity_score DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_resources():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT resource_type, available, total FROM resources")
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully with schema and dummy resources.")