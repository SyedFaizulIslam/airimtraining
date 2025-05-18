Feature: AI API Endpoints
  As a user
  I want to interact with various AI models through the API
  So that I can get responses for different use cases

  Scenario: Get chat response from OpenAI
    Given the API is running
    When I send a POST request to "/chat" with the prompt "Hello, how are you?"
    Then I should receive a successful response
    And the response should contain content from the model

  Scenario: Get QnA response from OpenAI via LangChain
    Given the API is running
    When I send a POST request to "/qna" with the prompt "What is machine learning?"
    Then I should receive a successful response
    And the response should contain a question and answer
    And the source should be "OpenAI via LangChain"

  Scenario: Predict consumer complaint category
    Given the API is running
    When I send a POST request to "/predictconsumercompaint" with the prompt "I had an issue with my credit card"
    Then I should receive a successful response
    And the response should contain a prediction category

  Scenario: Get response from fine-tuned QLora model
    Given the API is running
    When I send a POST request to "/qlora_fine_tune" with the prompt "Tell me about technology"
    Then I should receive a successful response
    And the response should contain generated text
