# text_utils.py
import re

def clean_text(text):
    """
    Applies some pre-processing on the given text.
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = text.strip().lower()
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    return text
