Feature: PDF Document Operations
  As a user of the AI API
  I want to upload PDF documents and query information from them
  So that I can extract insights from my documents using AI

  Background:
    Given the API is running

  Scenario: Upload a PDF document for embedding
    When I upload a PDF file to the "/Demo_Pdf_embeddings" endpoint
    Then I should receive a successful response
    And the response should contain the uploaded PDF filename
    And the response should indicate successful embedding
    And the response should include embedding details

  Scenario: Query information from an embedded PDF
    Given I have uploaded a PDF document
    When I send a question "What is the main topic of the document?" to the "/demo_query_pdf" endpoint
    Then I should receive a successful response
    And the response should contain the original question
    And the response should contain an answer based on the document
    And the response should include source information

  Scenario: Search for specific content in embedded PDF
    Given I have uploaded a PDF document
    When I search for "important" in the PDF using the "/search_pdf" endpoint
    Then I should receive a successful response
    And the response should contain matching document chunks
    And each result should include document metadata
