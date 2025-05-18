Feature: String Manipulation
  As a user
  I want to manipulate strings
  So that I can reverse strings and count characters

  Scenario: Reversing a string
    Given I have a StringManipulator
    When I reverse the string "hello"
    Then the result should be "olleh"

  Scenario: Reversing an empty string
    Given I have a StringManipulator
    When I reverse the string ""
    Then the result should be ""

  Scenario: Reversing a palindrome
    Given I have a StringManipulator
    When I reverse the string "radar"
    Then the result should be "radar"

  Scenario: Counting characters in a string
    Given I have a StringManipulator
    When I count characters in "hello world"
    Then the character count should be 11

  Scenario: Counting characters in an empty string
    Given I have a StringManipulator
    When I count characters in ""
    Then the character count should be 0

  Scenario: Error handling for non-string input in reverse method
    Given I have a StringManipulator
    When I try to reverse a non-string value
    Then it should raise a TypeError

  Scenario: Error handling for non-string input in count method
    Given I have a StringManipulator
    When I try to count characters in a non-string value
    Then it should raise a TypeError
