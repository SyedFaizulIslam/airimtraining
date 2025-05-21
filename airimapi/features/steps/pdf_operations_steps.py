from behave import given, when, then
import requests
import os
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_PDF_PATH = str(Path(__file__).parent.parent / "test_files" / "sample.pdf")

@given("I have uploaded a PDF document")
def step_impl_upload_pdf(context):
    # First ensure we have a test PDF file
    if not os.path.exists(TEST_PDF_PATH):
        # Create the test_files directory if it doesn't exist
        os.makedirs(os.path.dirname(TEST_PDF_PATH), exist_ok=True)
        
        # Create a simple PDF file for testing if needed
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(TEST_PDF_PATH)
            c.drawString(100, 750, "This is a test PDF document for acceptance testing.")
            c.drawString(100, 730, "It contains sample text that will be used for embeddings.")
            c.drawString(100, 710, "The main topic of this document is acceptance testing.")
            c.drawString(100, 690, "This document contains important information about PDF operations.")
            c.save()
        except ImportError:
            # If reportlab is not available, create a simple text file
            with open(TEST_PDF_PATH.replace('.pdf', '.txt'), 'w') as f:
                f.write("This is a test document for acceptance testing.\n")
                f.write("It contains sample text that will be used for embeddings.\n")
                f.write("The main topic of this document is acceptance testing.\n")
                f.write("This document contains important information about PDF operations.\n")
            # Alert the user that we couldn't create a proper PDF
            print("WARNING: ReportLab not installed. Created a text file instead of PDF.")
            print("Please install ReportLab with: pip install reportlab")
            TEST_PDF_PATH = TEST_PDF_PATH.replace('.pdf', '.txt')
    
    # Make sure context.api_base_url is set
    if not hasattr(context, 'api_base_url'):
        context.api_base_url = API_BASE_URL
    
    # Upload the PDF file
    with open(TEST_PDF_PATH, 'rb') as f:
        files = {'file': (os.path.basename(TEST_PDF_PATH), f, 'application/pdf' if TEST_PDF_PATH.endswith('.pdf') else 'text/plain')}
        response = requests.post(
            f"{context.api_base_url}/Demo_Pdf_embeddings",
            files=files
        )
    
    assert response.status_code == 200, f"PDF upload failed: {response.text}"
    
    response_data = response.json()
    # Store collection name for later queries
    context.collection_name = response_data.get('details', {}).get('collection_name', 'default')
    context.uploaded_filename = response_data.get('details', {}).get('filename', 'unknown')

@when("I upload a PDF file to the \"{endpoint}\" endpoint")
def step_impl(context, endpoint):
    # Ensure we have a test PDF file
    if not os.path.exists(TEST_PDF_PATH):
        assert False, f"Test PDF file not found at {TEST_PDF_PATH}"
    
    # Upload the PDF file
    with open(TEST_PDF_PATH, 'rb') as f:
        files = {'file': (os.path.basename(TEST_PDF_PATH), f, 'application/pdf' if TEST_PDF_PATH.endswith('.pdf') else 'text/plain')}
        context.response = requests.post(
            f"{context.api_base_url}{endpoint}",
            files=files
        )

@when("I send a question \"{question}\" to the \"{endpoint}\" endpoint")
def step_impl(context, question, endpoint):
    # Prepare the payload
    payload = {
        "prompt": question
    }
    
    # Add collection name if we have it
    params = {}
    if hasattr(context, 'collection_name'):
        params['collection_name'] = context.collection_name
    
    # Send the request
    context.response = requests.post(
        f"{context.api_base_url}{endpoint}",
        json=payload,
        params=params
    )

@when("I search for \"{query}\" in the PDF using the \"{endpoint}\" endpoint")
def step_impl(context, query, endpoint):
    # Prepare the payload
    payload = {
        "prompt": query
    }
    
    # Add collection name if we have it
    params = {}
    if hasattr(context, 'collection_name'):
        params['collection_name'] = context.collection_name
    
    # Send the request
    context.response = requests.post(
        f"{context.api_base_url}{endpoint}",
        json=payload,
        params=params
    )

@then("I should receive a successful response")
def step_impl(context):
    assert context.response.status_code == 200, f"Expected status code 200, got {context.response.status_code}. Response: {context.response.text}"

@then("the response should contain the uploaded PDF filename")
def step_impl(context):
    response_data = context.response.json()
    # Check in different places where the filename might be
    filename_found = False
    
    if 'filename' in response_data:
        filename_found = True
    elif 'details' in response_data and 'filename' in response_data['details']:
        filename_found = True
    
    assert filename_found, f"Response doesn't contain filename: {response_data}"

@then("the response should indicate successful embedding")
def step_impl(context):
    response_data = context.response.json()
    assert 'status' in response_data, f"Response doesn't contain status field: {response_data}"
    assert response_data['status'] == 'success', f"Status is not 'success': {response_data['status']}"

@then("the response should include embedding details")
def step_impl(context):
    response_data = context.response.json()
    assert 'details' in response_data, f"Response doesn't contain details: {response_data}"
    
    details = response_data['details']
    # Check for key embedding details
    essential_fields = ['chunks_created', 'collection_name', 'embedding_model']
    for field in essential_fields:
        assert field in details, f"Details missing '{field}': {details}"

@then("the response should contain the original question")
def step_impl(context):
    response_data = context.response.json()
    assert 'question' in response_data, f"Response doesn't contain question field: {response_data}"

@then("the response should contain an answer based on the document")
def step_impl(context):
    response_data = context.response.json()
    assert 'answer' in response_data, f"Response doesn't contain answer field: {response_data}"
    assert len(response_data['answer']) > 10, f"Answer seems too short: {response_data['answer']}"

@then("the response should include source information")
def step_impl(context):
    response_data = context.response.json()
    assert 'sources' in response_data, f"Response doesn't contain sources field: {response_data}"
    # Even if empty, it should be a list
    assert isinstance(response_data['sources'], list), f"Sources is not a list: {response_data['sources']}"

@then("the response should contain matching document chunks")
def step_impl(context):
    response_data = context.response.json()
    assert 'results' in response_data, f"Response doesn't contain results field: {response_data}"
    assert len(response_data['results']) > 0, "No matching document chunks found"

@then("each result should include document metadata")
def step_impl(context):
    response_data = context.response.json()
    results = response_data.get('results', [])
    
    for result in results:
        assert 'metadata' in result, f"Result missing metadata: {result}"
        metadata = result['metadata']
        assert 'filename' in metadata, f"Metadata missing filename: {metadata}"
