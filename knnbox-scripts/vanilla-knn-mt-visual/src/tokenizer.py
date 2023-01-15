import jieba
import sacremoses

# jieba
def jieba_tokenize(str):
    str = str.strip()
    str = jieba.cut(str)
    str = " ".join(str)
    return str

# space
def space_tokenize(str):
    str = str.strip()
    return str

# moses
def _mosestokenize(str, lang):
    mt = sacremoses.MosesTokenizer(lang=lang)
    str = str.strip()
    str = mt.tokenize(str, return_str=True) 
    return str

def moses_de_tokenize(str):
    return _mosestokenize(str, "de")

def moses_en_tokenize(str):
   return _mosetokenize(str, "en")

# Regist your function here
TOKENIZER_FUNCTIONS = {
    "jieba": jieba_tokenize,
    "moses-en": moses_en_tokenize,
    "moses-de": moses_de_tokenize,
    "space": space_tokenize,
}

