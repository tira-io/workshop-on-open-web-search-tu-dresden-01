import spacy
SPACY_MODELS = {"english": "en_core_web_sm"}

def lemmatize_text(text, language, nlp):
    
    tokens = []

    for token in nlp(text):
        if not token.is_stop:
            tokens.append((token.lemma_.lower()))

    return " ".join(tokens)