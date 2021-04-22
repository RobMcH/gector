import spacy
import random
import numpy as np
# https://www.nodebox.net/code/index.php/Linguistics.html
import nodebox_linguistics_extended as nle

# Used for obtaining POS tags.
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
# Pronoun lists #
SBJ_PRONOUNS = {"i", "you", "he", "she", "it", "we", "you", "they"}
OBJ_PRONOUNS = {"me", "you", "him", "her", "it", "us", "you", "them"}
POS_PRONOUNS = {"mine", "yours", "his", "hers", "ours", "yours", "theirs"}
REF_PRONOUNS = {"myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves"}
COMPLEMENT_PRONOUNS = {0: [OBJ_PRONOUNS, POS_PRONOUNS, REF_PRONOUNS],
                       1: [SBJ_PRONOUNS, POS_PRONOUNS, REF_PRONOUNS],
                       2: [SBJ_PRONOUNS, OBJ_PRONOUNS, REF_PRONOUNS],
                       3: [SBJ_PRONOUNS, OBJ_PRONOUNS, POS_PRONOUNS]}
# --- --- #


def find_word_perturbation(sentence: str, target_idx: int) -> str:
    doc = nlp(sentence)
    target_token = doc[target_idx]
    pos = target_token.pos_
    if pos == "NOUN":
        perturbation = perturb_noun(target_token)
    elif pos == "VERB":
        perturbation = perturb_verb(target_token)
    elif pos == "ADJ":
        perturbation = perturb_adjective(target_token)
    elif pos == "ADV":
        perturbation = perturb_adverb(target_token)
    elif pos == "PRON":
        perturbation = perturb_pronoun(target_token)
    elif pos == "NUM" or pos == "PROPN":
        # Remain unchanged.
        perturbation = target_token.text
    elif pos == "PUNCT":
        # Delete punctuation.
        perturbation = ""
    else:
        # Delete or UNK.
        perturbation = perturb(target_token)
    return doc.text.replace(target_token.text, perturbation)


def perturb_noun(token: spacy.tokens.token.Token) -> str:
    # Change singular nouns to plural nouns and vice versa.
    if token.tag_ == "NNS":
        # Plural noun, convert to singular.
        return nle.noun.singular(token.text)
    elif token.tag_ == "NN":
        # Singular noun, convert to plural.
        return nle.noun.plural(token.text)


def perturb_verb(token: spacy.tokens.token.Token) -> str:
    # Perturb verb by changing it to its present, progressive, past, perfect, or 3rd person singular form.
    form = np.random.randint(5)
    if form == 0:
        # Present form.
        return nle.verb.present(token.text)
    elif form == 1:
        # Progressive form.
        return nle.verb.present_participle(token.text)
    elif form == 2:
        # Past form.
        return nle.verb.past(token.text)
    elif form == 3:
        # Perfect form.
        return nle.verb.past_participle(token.text)
    elif form == 4:
        # 3rd person singular form.
        return nle.verb.present(token.text, person=3)


def perturb_adjective(token: spacy.tokens.token.Token) -> str:
    # Convert adjective to adverb. Ignoring irregular cases.
    if token.text[-1] == "y":
        return f"{token.text[:-1]}ily"
    elif token.text[-4:] == "able" or token.text[-4:] == "ible" or token.text[-2:] == "le":
        return f"{token.text[:-1]}y"
    elif token.text[-2:] == "ic":
        return f"{token.text}ally"
    else:
        return f"{token.text}ly"


def perturb_adverb(token: spacy.tokens.token.Token) -> str:
    # Convert adverb to adjective. Ignoring irregular cases.
    if token.text[-3:] == "ily":
        return f"{token.text[:-3]}y"
    elif token.text[-3:] == "bly":
        return f"{token.text[:-1]}e"
    elif token.text[-6:] == "ically":
        return token.text[:-4]
    else:
        return token.text[:-2]


def perturb_pronoun(token: spacy.tokens.token.Token) -> str:
    pronoun = token.text.lower()
    if pronoun in SBJ_PRONOUNS:
        pronoun_category = 0
    elif pronoun in OBJ_PRONOUNS:
        pronoun_category = 1
    elif pronoun in POS_PRONOUNS:
        pronoun_category = 2
    elif pronoun in REF_PRONOUNS:
        pronoun_category = 3
    else:
        # Error handling.
        return token.text
    # Randomly pick one of the three complement categories.
    sample_category = np.random.randint(3)
    # Randomly pick one of the forms in this category.
    return random.sample(COMPLEMENT_PRONOUNS[pronoun_category][sample_category], 1)[0]


def perturb(token: spacy.tokens.token.Token) -> str:
    # Perturb word by removing token or replacing it with UNK symbol.
    if np.random.uniform < 0.5:
        # Delete token.
        return ""
    else:
        return "<UNK>"
