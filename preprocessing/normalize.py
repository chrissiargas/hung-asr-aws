import re
import unicodedata

def normalize_symbols(text):
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')

    return text

def normalize(text):
    if not text:
        return ''

    text = text.lower()
    text = normalize_symbols(text)

    # 2. Remove speaker tags at the beginning (e.g. ομιλητής 1: ")
    text = re.sub(r'^\s*[\w\s\.\-]{1,30}:\s*', '', text)

    # Remove acoustic tags like <spoken_noise>, <unk>
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove transcriber notes in parentheses or brackets e.g., (Χειροκροτήματα)
    text = re.sub(r'\[.*?\]|\(.*?\)', ' ', text)

    # Collapse multiple spaces into a single space and strip edges
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'([^\w\s])\1*', r' \g<0> ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

if __name__ == "__main__":
    sentence = 'Έχω ένα μικρό... Ελεφαντάκι!'
    print(normalize(sentence, with_signs=False))
    print(normalize(sentence, with_signs=True))