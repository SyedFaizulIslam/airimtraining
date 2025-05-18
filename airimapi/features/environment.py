import os
import subprocess
import time
import signal
import sys
import requests
import psutil  # Add this package to requirements-test.txt

# Global variables
server_process = None

def before_all(context):
    # Start the server only once before all tests
    start_server(context)
    
def after_all(context):
    # Stop the server after all tests
    stop_server(context)
    
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
