import gzip
import re
import sqlite3
import os
import time
import csv
from tqdm import tqdm

# Regex patterns to extract Namespace 0 records from INSERT statements
# page: (page_id, page_namespace, 'page_title', page_is_redirect, page_is_new, ...)
PAGE_RE = re.compile(r"\((\d+),0,'((?:[^'\\]|\\.)*)',([01]),([01])")
# linktarget: (lt_id, lt_namespace, 'lt_title')
LINKTARGET_RE = re.compile(r"\((\d+),0,'((?:[^'\\]|\\.)*)'\)")
# pagelinks: (pl_from, pl_from_namespace, pl_target_id)
# Note: we only care about links originating from namespace 0 (source_ns = 0)
LINK_RE = re.compile(r"\((\d+),0,(\d+)\)")

class ProgressFileWrapper:
    """Wraps a file object to automatically update a tqdm progress bar on read."""
    def __init__(self, filepath, desc="Parsing"):
        self.file = open(filepath, 'rb')
        self.total = os.path.getsize(filepath)
        self.pbar = tqdm(total=self.total, unit='B', unit_scale=True, desc=desc)
    
    def read(self, size=-1):
        data = self.file.read(size)
        self.pbar.update(len(data))
        return data
        
    def seek(self, offset, whence=0):
        return self.file.seek(offset, whence)
        
    def tell(self):
        return self.file.tell()
    
    def close(self):
        self.pbar.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def init_build_status(cursor):
    """Initializes the build_status table and infers state for existing databases."""
    cursor.execute("CREATE TABLE IF NOT EXISTS build_status (step TEXT PRIMARY KEY, status TEXT)")
    
    # Migration for existing databases
    cursor.execute("SELECT count(*) FROM build_status")
    if cursor.fetchone()[0] == 0:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        if 'pages' in tables:
            cursor.execute("SELECT count(*) FROM pages")
            if cursor.fetchone()[0] > 0:
                cursor.execute("INSERT INTO build_status VALUES ('pages', 'DONE')")
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_pages_title'")
                if cursor.fetchone():
                    cursor.execute("INSERT INTO build_status VALUES ('index_pages', 'DONE')")
                    
        if 'linktarget' in tables:
            cursor.execute("SELECT count(*) FROM linktarget")
            if cursor.fetchone()[0] > 0:
                cursor.execute("INSERT INTO build_status VALUES ('linktarget', 'DONE')")
                
        if 'temp_links' in tables:
            cursor.execute("SELECT count(*) FROM temp_links")
            if cursor.fetchone()[0] > 0:
                cursor.execute("INSERT INTO build_status VALUES ('pagelinks', 'DONE')")
                
        if 'links' in tables and 'temp_links' not in tables:
            cursor.execute("SELECT count(*) FROM links")
            if cursor.fetchone()[0] > 0:
                cursor.execute("INSERT INTO build_status VALUES ('resolve_links', 'DONE')")
                cursor.execute("INSERT INTO build_status VALUES ('cleanup', 'DONE')")

def is_done(cursor, step):
    cursor.execute("SELECT status FROM build_status WHERE step=?", (step,))
    row = cursor.fetchone()
    return row is not None and row[0] == 'DONE'

def mark_done(cursor, step):
    cursor.execute("INSERT OR REPLACE INTO build_status VALUES (?, 'DONE')", (step,))

def parse_and_load(db_path, page_sql_path, linktarget_sql_path, links_sql_path):
    # Ensure data directory for DB exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable SQLite speed optimizations but respect RAM limits
    cursor.execute("PRAGMA synchronous = NORMAL")
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA cache_size = -512000")   # 512 MB page cache (safe for most systems)
    cursor.execute("PRAGMA temp_store = MEMORY")    # Use RAM for temp tables for maximum speed
    cursor.execute("PRAGMA mmap_size = 8589934592") # 8 GB memory-mapped I/O for fast reads (mmap is lazy, doesn't reserve RAM)
    
    # Initialize robust state tracking
    init_build_status(cursor)
    conn.commit()
    
    # --- Step 1: Pages ---
    if not is_done(cursor, 'pages'):
        print("\n[Step 1/5] Parsing page dump...")
        cursor.execute("DROP TABLE IF EXISTS pages")
        cursor.execute("""
        CREATE TABLE pages (
            id INTEGER PRIMARY KEY,
            title TEXT,
            is_redirect INTEGER
        )""")
        conn.commit()
        
        t_start = time.time()
        records = []
        with ProgressFileWrapper(page_sql_path, desc="Pages") as wrapper:
            with gzip.GzipFile(fileobj=wrapper, mode='rb') as gz:
                import io
                f = io.TextIOWrapper(gz, encoding="utf-8", errors="ignore")
                for line in f:
                    if not line.startswith("INSERT INTO `page` VALUES"):
                        continue
                    for m in PAGE_RE.findall(line):
                        records.append((int(m[0]), m[1], int(m[2])))
                        if len(records) >= 500000:
                            cursor.executemany("INSERT INTO pages VALUES (?,?,?)", records)
                            conn.commit()
                            records = []
        if records:
            cursor.executemany("INSERT INTO pages VALUES (?,?,?)", records)
            conn.commit()
        mark_done(cursor, 'pages')
        conn.commit()
        print(f"Pages populated in {time.time() - t_start:.2f}s")
    else:
        print("\n[Step 1/5] Pages already parsed. Skipping.")

    # --- Step 2: Index Pages ---
    if not is_done(cursor, 'index_pages'):
        print("\n[Step 2/5] Creating index on pages.title...")
        t_index = time.time()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_title ON pages(title)")
        mark_done(cursor, 'index_pages')
        conn.commit()
        print(f"Index created in {time.time() - t_index:.2f}s")
    else:
        print("\n[Step 2/5] Pages index already exists. Skipping.")

    # --- Step 3: Linktargets ---
    if not is_done(cursor, 'linktarget'):
        print("\n[Step 3/5] Parsing linktarget dump...")
        cursor.execute("DROP TABLE IF EXISTS linktarget")
        cursor.execute("""
        CREATE TABLE linktarget (
            id INTEGER PRIMARY KEY,
            title TEXT
        )""")
        conn.commit()
        
        t_start = time.time()
        records = []
        with ProgressFileWrapper(linktarget_sql_path, desc="Linktargets") as wrapper:
            with gzip.GzipFile(fileobj=wrapper, mode='rb') as gz:
                import io
                f = io.TextIOWrapper(gz, encoding="utf-8", errors="ignore")
                for line in f:
                    if not line.startswith("INSERT INTO `linktarget` VALUES"):
                        continue
                    for m in LINKTARGET_RE.findall(line):
                        records.append((int(m[0]), m[1]))
                        if len(records) >= 1000000:  # Larger batch: 2-col table has lower per-row overhead
                            cursor.executemany("INSERT INTO linktarget VALUES (?,?)", records)
                            conn.commit()
                            records = []
        if records:
            cursor.executemany("INSERT INTO linktarget VALUES (?,?)", records)
            conn.commit()
        mark_done(cursor, 'linktarget')
        conn.commit()
        print(f"Linktargets populated in {time.time() - t_start:.2f}s")
    else:
        print("\n[Step 3/5] Linktargets already parsed. Skipping.")

    # --- Step 4: Pagelinks ---
    if not is_done(cursor, 'pagelinks'):
        print("\n[Step 4/5] Parsing pagelinks dump...")
        cursor.execute("DROP TABLE IF EXISTS temp_links")
        cursor.execute("""
        CREATE TABLE temp_links (
            source_id INTEGER,
            target_lt_id INTEGER
        )""")
        conn.commit()
        
        t_start = time.time()
        records = []
        with ProgressFileWrapper(links_sql_path, desc="Pagelinks") as wrapper:
            with gzip.GzipFile(fileobj=wrapper, mode='rb') as gz:
                import io
                f = io.TextIOWrapper(gz, encoding="utf-8", errors="ignore")
                for line in f:
                    if not line.startswith("INSERT INTO `pagelinks` VALUES"):
                        continue
                    for m in LINK_RE.findall(line):
                        records.append((int(m[0]), int(m[1])))
                        if len(records) >= 1000000:
                            cursor.executemany("INSERT INTO temp_links VALUES (?,?)", records)
                            conn.commit()
                            records = []
        if records:
            cursor.executemany("INSERT INTO temp_links VALUES (?,?)", records)
            conn.commit()
        mark_done(cursor, 'pagelinks')
        conn.commit()
        print(f"Pagelinks populated in {time.time() - t_start:.2f}s")
    else:
        print("\n[Step 4/5] Pagelinks already parsed. Skipping.")

    # --- Step 5: Resolve Links ---
    if not is_done(cursor, 'resolve_links'):
        print("\n[Step 5/5] Resolving links...")
        t_start = time.time()
        
        cursor.execute("DROP TABLE IF EXISTS links")
        cursor.execute("""
        CREATE TABLE links (
            source_id INTEGER,
            target_id INTEGER
        )""")
        
        # Aggressively limit memory for this phase:
        # - Disable mmap so SQLite doesn't pull the multi-GB DB file into resident RAM
        # - Force temp/sort spills to disk
        # - Use a moderate cache (512 MB)
        cursor.execute("PRAGMA mmap_size = 0")
        cursor.execute("PRAGMA temp_store = FILE")
        cursor.execute("PRAGMA cache_size = -512000")
        
        # Phase 1: Pre-resolve the string join ONCE on the smaller tables.
        # linktarget (~20-30M rows) × pages (~7M rows) is far more manageable than
        # repeating this string join for every chunk of the 500M+ temp_links table.
        print("  1. Building linktarget -> page_id mapping (string join, one-time)...")
        t_map = time.time()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lt_title ON linktarget(title)")
        conn.commit()
        
        cursor.execute("DROP TABLE IF EXISTS lt_to_page")
        cursor.execute("""
            CREATE TABLE lt_to_page (
                lt_id INTEGER PRIMARY KEY,
                page_id INTEGER
            )
        """)
        cursor.execute("""
            INSERT INTO lt_to_page (lt_id, page_id)
            SELECT lt.id, p.id
            FROM linktarget lt
            JOIN pages p ON lt.title = p.title
        """)
        conn.commit()
        map_count = cursor.rowcount if cursor.rowcount > 0 else 0
        print(f"     Mapping built in {time.time() - t_map:.2f}s ({map_count:,} entries)")
        
        # Drop the string index — no longer needed
        cursor.execute("DROP INDEX IF EXISTS idx_lt_title")
        conn.commit()
        
        # Phase 2: Resolve temp_links via fast INTEGER join in chunks.
        # lt_to_page has an INTEGER PRIMARY KEY so lookups are O(log N) B-tree seeks.
        print("  2. Resolving links via integer join in chunks...")
        t_resolve = time.time()
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tl_target ON temp_links(target_lt_id)")
        conn.commit()
        
        cursor.execute("SELECT MIN(rowid), MAX(rowid) FROM temp_links")
        min_rowid, max_rowid = cursor.fetchone()
        
        chunk_size = 10_000_000
        resolved_total = 0
        for chunk_start in range(min_rowid, max_rowid + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size - 1, max_rowid)
            cursor.execute("""
                INSERT INTO links (source_id, target_id)
                SELECT tl.source_id, m.page_id
                FROM temp_links tl
                JOIN lt_to_page m ON tl.target_lt_id = m.lt_id
                WHERE tl.rowid BETWEEN ? AND ?
            """, (chunk_start, chunk_end))
            conn.commit()
            resolved_total += cursor.rowcount if cursor.rowcount > 0 else 0
            pct = (chunk_end - min_rowid + 1) / (max_rowid - min_rowid + 1) * 100
            print(f"     Progress: {pct:.1f}% ({resolved_total:,} links resolved)")
        
        print(f"     Resolved in {time.time() - t_resolve:.2f}s")
        
        # Cleanup mapping table and indexes
        cursor.execute("DROP TABLE IF EXISTS lt_to_page")
        cursor.execute("DROP INDEX IF EXISTS idx_tl_target")
        
        # Restore memory settings for subsequent operations
        cursor.execute("PRAGMA temp_store = MEMORY")
        cursor.execute("PRAGMA mmap_size = 8589934592")
        
        mark_done(cursor, 'resolve_links')
        conn.commit()
        print(f"Links resolved in {time.time() - t_start:.2f}s")
    else:
        print("\n[Step 5/5] Links already resolved. Skipping.")

    # --- Cleanup ---
    if not is_done(cursor, 'cleanup'):
        print("\nCleaning up temporary database tables...")
        cursor.execute("DROP TABLE IF EXISTS temp_links")
        cursor.execute("DROP TABLE IF EXISTS linktarget")
        conn.commit()
        
        print("Running database VACUUM...")
        cursor.execute("PRAGMA temp_store = FILE") # Force temp files to disk so VACUUM doesn't hold the whole DB in RAM
        cursor.execute("VACUUM")
        mark_done(cursor, 'cleanup')
        conn.commit()
        print("\nDatabase built successfully!")
    else:
        print("\nDatabase cleanup already performed.")
        
    conn.close()

def export_edges_to_file(db_path, edge_list_path):
    """Export resolved links from SQLite database to a space-separated text file."""
    print(f"\n[Step 2] Exporting edges from {db_path} to {edge_list_path}...")
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Read-optimized PRAGMAs for sequential full-table scan
    cursor.execute("PRAGMA mmap_size = 8589934592")  # 8 GB mmap for fast sequential reads (lazy, no upfront RAM)
    cursor.execute("PRAGMA cache_size = -256000")    # 256 MB page cache for read buffer
    
    cursor.execute("SELECT source_id, target_id FROM links")
    
    os.makedirs(os.path.dirname(edge_list_path), exist_ok=True)
    with open(edge_list_path, "w", encoding="utf-8", newline='') as f:
        # csv.writer is implemented in C and is massively faster than Python f-strings
        writer = csv.writer(f, delimiter=' ')
        while True:
            rows = cursor.fetchmany(1000000)
            if not rows:
                break
            writer.writerows(rows)
            
    conn.close()
    print(f"Edges exported successfully in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    db_path = r"data\wiki_graph.db"
    working_db_path = r"data\wiki_graph_working.db"
    page_sql_path = r"data\enwiki-20260501-page.sql.gz"
    linktarget_sql_path = r"data\enwiki-20260501-linktarget.sql.gz"
    links_sql_path = r"data\enwiki-20260501-pagelinks.sql.gz"
    edge_list_path = r"data\edge_list.txt"
    
    # Ensure working DB exists and is fully built
    if os.path.exists(db_path) and not os.path.exists(working_db_path):
        import shutil
        print(f"Copying existing DB {db_path} to working copy {working_db_path}...")
        shutil.copy2(db_path, working_db_path)
        
    # parse_and_load now natively handles partial progress and gracefully skips completed steps
    parse_and_load(working_db_path, page_sql_path, linktarget_sql_path, links_sql_path)
    
    # Run Step 2: Export SQLite links to edge list text file
    if not os.path.exists(edge_list_path):
        export_edges_to_file(working_db_path, edge_list_path)
    else:
        print(f"Edge list file {edge_list_path} already exists. Skipping Step 2 export.")
