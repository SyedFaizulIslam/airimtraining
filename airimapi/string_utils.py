class StringManipulator:
    """
    A class that provides string manipulation utilities.
    """
    
    def reverse_string(self, input_string):
        """
        Reverses the input string.
        
        Args:
            input_string (str): The string to be reversed
            
        Returns:
            str: The reversed string
        """
        if not isinstance(input_string, str):
            raise TypeError("Input must be a string")
        
        return input_string[::-1]
    
    def count_characters(self, input_string):
        """
        Counts the number of characters in the input string.
        
        Args:
            input_string (str): The string to count characters in
            
        Returns:
            int: The number of characters in the string
        """
        if not isinstance(input_string, str):
            raise TypeError("Input must be a string")
        
        return len(input_string)
