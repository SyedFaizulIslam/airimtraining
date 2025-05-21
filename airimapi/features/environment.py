import os
import subprocess
import time
import signal
import sys
import requests
from pathlib import Path
import psutil  # Add this package to requirements-test.txt

# Global variables
server_process = None

def before_all(context):
    # Start the server only once before all tests
    start_server(context)
    
    # Create test files directory and sample PDF if needed
    create_test_files(context)
    
def after_all(context):
    # Stop the server after all tests
    stop_server(context)

def create_test_files(context):
    # Create test files directory
    test_files_dir = Path(__file__).parent / "test_files"
    os.makedirs(test_files_dir, exist_ok=True)
    
    # Create a sample PDF for testing if it doesn't exist
    sample_pdf_path = test_files_dir / "sample.pdf"
    
    if not sample_pdf_path.exists():
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            print(f"Creating sample PDF at {sample_pdf_path}")
            c = canvas.Canvas(str(sample_pdf_path), pagesize=letter)
            c.drawString(100, 750, "This is a sample PDF document for testing.")
            c.drawString(100, 730, "It contains various terms that can be searched.")
            c.drawString(100, 710, "The main topic of this document is PDF API testing.")
            c.drawString(100, 690, "This document contains important information about embeddings.")
            c.drawString(100, 670, "The data can be processed and queried via the API.")
            c.save()
            print("Sample PDF created successfully.")
        except ImportError:
            print("Warning: ReportLab not installed. Unable to create sample PDF.")
            print("Please install reportlab with: pip install reportlab")
            
            # Create a text file as fallback
            with open(str(sample_pdf_path).replace('.pdf', '.txt'), 'w') as f:
                f.write("This is a sample document for testing.\n")
                f.write("It contains various terms that can be searched.\n")
                f.write("The main topic of this document is PDF API testing.\n")
                f.write("This document contains important information about embeddings.\n")
                f.write("The data can be processed and queried via the API.\n")
                
            print("Created text file instead as fallback.")

def start_server(context):
    global server_process
    
    # Check if server is already running
    try:
        response = requests.get("http://localhost:8000/docs")
        if response.status_code == 200:
            print("Server is already running, using existing instance.")
            context.server_started = True
            return
    except:
        pass
    
    # Get the directory of the main.py file
    main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
    
    print(f"Starting server using: {sys.executable} {main_path}")
    
    # Start the server in a subprocess
    try:
        # Ensure we're running in a shell on Windows
        use_shell = sys.platform.startswith('win')
        
        server_process = subprocess.Popen(
            [sys.executable, main_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=use_shell
        )
        
        # Wait for server to start
        print("Waiting for server to start...")
        for attempt in range(15):  # Increased to 15 attempts
            time.sleep(2)  # Increased wait time between attempts
            try:
                response = requests.get("http://localhost:8000/docs")
                if response.status_code == 200:
                    print(f"Server started successfully after {attempt+1} attempts!")
                    context.server_started = True
                    return
            except requests.ConnectionError:
                print(f"Attempt {attempt+1}: Server not ready yet...")
                continue
        
        print("Failed to start server after multiple attempts")
        if server_process:
            # Print any output from the server process
            stdout, stderr = server_process.communicate(timeout=1)
            print("Server stdout:", stdout.decode())
            print("Server stderr:", stderr.decode())
        raise Exception("Server failed to start within the timeout period")
    
    except Exception as e:
        print(f"Error starting server: {e}")
        if server_process:
            server_process.terminate()
        raise

def stop_server(context):
    global server_process
    if server_process:
        print("Stopping server...")
        pid = server_process.pid
        
        try:
            # On Windows, we need to kill the process group
            if sys.platform.startswith('win'):
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
            else:
                # On Unix-like systems
                server_process.terminate()
                
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Server didn't terminate gracefully, force killing...")
                if sys.platform.startswith('win'):
                    parent = psutil.Process(pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                else:
                    server_process.kill()
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass  # Process already terminated
            
        server_process = None
        print("Server stopped.")
