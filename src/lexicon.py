# src/lexicon.py
FIRST_PERSON = {
    "i", "me", "my", "mine", "myself",
}
NEGATIONS = {
    "no", "not", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "n't",
}
SADNESS = {
    "sad", "depressed", "lonely", "hopeless", "worthless",
    "cry", "crying", "tears", "empty", "numb",
}

def is_lexicon_token(tok_lower: str) -> bool:
    return (tok_lower in FIRST_PERSON) or (tok_lower in NEGATIONS) or (tok_lower in SADNESS)
