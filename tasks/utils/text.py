import re
import hashlib
from io import BytesIO
import ftfy
from cleantext import clean

def clean_extra_whitespaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)

def clean_broken_unicode(text: str) -> str:
    """
    Corrige problemas de codificação Unicode em uma string.

    Esta função utiliza a biblioteca ftfy (Fixes Text For You) para detectar e corrigir
    texto corrompido por erros de codificação, como 'mojibake'. O resultado é uma string
    onde os caracteres foram restaurados para sua forma correta.

    Parâmetros:
    text (str): A string de entrada que pode conter caracteres Unicode corrompidos.

    Retorna:
    str: A string com os caracteres Unicode corrigidos.

    Exemplo:
    >>> clean_broken_unicode('Ã©')
    'é'
    
    """
    return ftfy.fix_text(text)
def clean_extracted_text(text: str) -> str:
    cleaned_text = clean(
        text, 
        fix_unicode=True, 
        to_ascii=True, 
        lang="pt"
    )

    return cleaned_text

def get_checksum(source_text: str) -> str:
    """Calculate the md5 checksum of text
    by creating a file-like object without reading its
    whole content in memory.

    Example
    -------
    >>> extractor.get_checksum("A simple text")
        'ef313f200597d0a1749533ba6aeb002e'
    """
    file = BytesIO(source_text.encode(encoding="UTF-8"))

    m = hashlib.md5()
    while True:
        d = file.read(8096)
        if not d:
            break
        m.update(d)
    return m.hexdigest()