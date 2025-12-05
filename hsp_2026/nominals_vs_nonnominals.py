import pandas as pd
import spacy
import spacy.attrs
from tqdm import tqdm
import os

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
tqdm.pandas()

# Merge to get rep_num
df = pd.read_csv("for_pos_tag.csv")

# Function to classify PoS tokens
def classify_tokens(text):
    doc = nlp(str(text))
    noun = [token for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    modifier = [token for token in doc if token.pos_ in ["ADJ", "ADV"]]
    verb = [token for token in doc if token.pos_ in ["VERB"]]
    pro = [token for token in doc if token.pos_ in ["PRON"]]
    closed = [token for token in doc if token.pos_ in ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "SCONJ"]]
    return len(noun), len(modifier), len(verb), len(pro), len(closed)

# Apply classification with progress
df[["nouns", "modifiers", "verbs", "pro", "closed"]] = df["text"].fillna("").progress_apply(
    lambda x: pd.Series(classify_tokens(x))
)

df.to_csv("pos_tag.csv")