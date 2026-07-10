import re
import unicodedata
from num2words import num2words

def normalize_symbols(text):
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('„', '"').replace('”', '"')
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')

    return text

def verbalize_hungarian_numbers(text):
    def replace_match(match):
        num_str = match.group(0)
        try:
            return num2words(int(num_str), lang='hu')
        except ValueError:
            return num_str

    return re.sub(r'\d+', replace_match, text)

def clean_whitespaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize(text, with_signs=True):
    if not text:
        return ''

    text = unicodedata.normalize('NFC', text)
    text = normalize_symbols(text)

    # print(text)

    # 2. Remove speaker tags at the beginning (e.g. ομιλητής 1: ")
    text = re.sub(r'^\s*[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ0-9\s\.\-]{1,30}:\s*', '', text)

    # print(text)

    # Remove cutoff words (e.g., "valami~")
    text = re.sub(r'~', '', text)

    # print(text)

    # Remove transcriber notes in parentheses or brackets e.g., (Χειροκροτήματα)
    text = re.sub(r'\[.*?\]|\(\(.*?\)\)|\(.*?\)|<[^>]+>', ' ', text)

    # print(text)

    # This safely deletes any stray (, ), <, >, [, or ] left in the string
    text = re.sub(r'[()<>\[\]]', '', text)

    # print(text)

    text = verbalize_hungarian_numbers(text)

    # print(text)

    text = re.sub(r'#\s*\w*', '', text)

    # print(text)

    text = re.sub(r'^\s*:\s*', '', text)

    # print(text)

    if with_signs:
        text = re.sub(r'([^\w\s])\1*', r' \g<0> ', text)
    else:
        text = text.lower()
        text = re.sub(r'[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ\s]', ' ', text)

    # print(text)

    return clean_whitespaces(text)

if __name__ == "__main__":
    sentence = "Helen: ((Szia)) (! <lang:en>Hello, how are you?</lang:en> [Hogy] vagy? 132 ] el~ gr #ah # : #”"
    print(sentence)
    sentence_ = normalize(sentence)
    print(sentence_)
    sentence__ = normalize(sentence_)
    print(sentence__)
    print(sentence__ == sentence_)
