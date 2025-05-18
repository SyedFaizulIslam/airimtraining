from behave import given, when, then
import requests
import json
import time
import os
import subprocess
import signal
import atexit
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store the server process globally
server_process = None

def setup_server():
    global server_process
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main_path = os.path.join(current_dir, "main.py")
    
    # Start the server as a subprocess
    print("Starting FastAPI server...")
    server_process = subprocess.Popen([sys.executable, main_path],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
    
    # Register cleanup function to terminate server on exit
    atexit.register(lambda: server_process.terminate() if server_process else None)
    
    # Give the server time to start
    time.sleep(5)
    print("Server should be running now.")

def check_server_running():
    # Try to connect to the server to see if it's running
    try:
        response = requests.get("http://localhost:8000/docs")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

@given("the API is running")
def step_impl(context):
    # Verify the server is running - this should be handled by environment.py
    try:
        response = requests.get("http://localhost:8000/docs")
        assert response.status_code == 200, "FastAPI server is not running"
        logger.info("Server is running correctly")
    except requests.ConnectionError as e:
        logger.error(f"Server connection failed: {e}")
        assert False, "FastAPI server is not accessible"

@when('I send a POST request to "{endpoint}" with the prompt "{prompt}"')
def step_impl(context, endpoint, prompt):
    url = f"http://localhost:8000{endpoint}"
    payload = {"prompt": prompt}
    
    logger.info(f"Sending request to {url} with payload: {payload}")
    
    try:
        context.response = requests.post(url, json=payload)
        logger.info(f"Response status: {context.response.status_code}")
        logger.info(f"Response content: {context.response.text[:100]}...")  # Log first 100 chars
    except Exception as e:
        logger.error(f"Request failed: {e}")
        assert False, f"Failed to send request: {e}"

@then("I should receive a successful response")
def step_impl(context):
    assert context.response.status_code == 200, f"Expected 200, got {context.response.status_code}"

@then("the response should contain content from the model")
def step_impl(context):
    try:
        response_data = context.response.json()
        logger.info(f"Checking for content in: {response_data}")
        assert "content" in response_data, f"Response does not contain content field: {response_data}"
        assert response_data["content"], "Content field is empty"
    except json.JSONDecodeError:
        # Handle non-JSON responses
        assert len(context.response.text) > 0, "Response is empty"
        logger.info("Response is not JSON but contains text")

@then("the response should contain a question and answer")
def step_impl(context):
    try:
        response_data = context.response.json()
        assert "question" in response_data, f"Response does not contain question field: {response_data}"
        assert "answer" in response_data, f"Response does not contain answer field: {response_data}"
        assert response_data["question"], "Question field is empty"
        assert response_data["answer"], "Answer field is empty"
    except json.JSONDecodeError:
        assert False, "Expected JSON response for QnA endpoint"

@then('the source should be "{source}"')
def step_impl(context, source):
    try:
        response_data = context.response.json()
        assert "source" in response_data, f"Response does not contain source field: {response_data}"
        assert response_data["source"] == source, f"Source is {response_data['source']}, not {source}"
    except json.JSONDecodeError:
        assert False, "Expected JSON response with source field"

@then("the response should contain a prediction category")
def step_impl(context):
    # For this endpoint, the response might be a simple string
    try:
        response_data = context.response.json()
        assert response_data is not None, "No prediction category returned"
    except json.JSONDecodeError:
        # If not JSON, check the raw response
        assert context.response.text and len(context.response.text) > 0, "No prediction category returned"
    
@then("the response should contain generated text")
def step_impl(context):
    response_text = context.response.text
    assert response_text and len(response_text) > 10, f"Response does not contain enough generated text: '{response_text}'"
