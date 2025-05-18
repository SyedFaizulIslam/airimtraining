from behave import given, when, then
import sys
import os

# Add the parent directory to sys.path to import StringManipulator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from string_utils import StringManipulator

@given('I have a StringManipulator')
def step_impl(context):
    context.manipulator = StringManipulator()

@when('I reverse the string "{input_string}"')
def step_impl(context, input_string):
    context.result = context.manipulator.reverse_string(input_string)

@then('the result should be "{expected_result}"')
def step_impl(context, expected_result):
    assert context.result == expected_result, f"Expected '{expected_result}', got '{context.result}'"

@when('I count characters in "{input_string}"')
def step_impl(context, input_string):
    context.count = context.manipulator.count_characters(input_string)

@then('the character count should be {expected_count:d}')
def step_impl(context, expected_count):
    assert context.count == expected_count, f"Expected {expected_count}, got {context.count}"

@when('I try to reverse a non-string value')
def step_impl(context):
    context.exception = None
    try:
        context.manipulator.reverse_string(123)
    except Exception as e:
        context.exception = e

@when('I try to count characters in a non-string value')
def step_impl(context):
    context.exception = None
    try:
        context.manipulator.count_characters(123)
    except Exception as e:
        context.exception = e

@then('it should raise a TypeError')
def step_impl(context):
    assert isinstance(context.exception, TypeError), f"Expected TypeError, got {type(context.exception).__name__}"
