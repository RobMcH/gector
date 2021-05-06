import spacy
import random
import numpy as np
from typing import Tuple, Set
# https://www.nodebox.net/code/index.php/Linguistics.html
import nodebox_linguistics_extended as nle
from nltk.corpus import wordnet as wn

# Used for obtaining POS tags.
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
# Pronoun lists #
SBJ_PRONOUNS = ["i", "you", "he", "she", "it", "we", "you", "they"]
OBJ_PRONOUNS = ["me", "you", "him", "her", "it", "us", "you", "them"]
POS_PRONOUNS = ["mine", "yours", "his", "hers", "its", "ours", "yours", "theirs"]
REF_PRONOUNS = ["myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves"]
SBJ_SET, OBJ_SET, POS_SET, REF_SET = set(SBJ_PRONOUNS), set(OBJ_PRONOUNS), set(POS_PRONOUNS), set(REF_PRONOUNS)
COMPLEMENT_PRONOUNS = {0: [OBJ_PRONOUNS, POS_PRONOUNS, REF_PRONOUNS],
                       1: [SBJ_PRONOUNS, POS_PRONOUNS, REF_PRONOUNS],
                       2: [SBJ_PRONOUNS, OBJ_PRONOUNS, REF_PRONOUNS],
                       3: [SBJ_PRONOUNS, OBJ_PRONOUNS, POS_PRONOUNS]}
# --- --- #
# Verb tenses #
VERB_TENSES = {'present plural': 0, '1st singular present': 0, '2nd singular present': 0, 'present participle': 1,
               'past': 2, 'past plural': 2, '1st singular past': 2, '2nd singular past': 2, '3rd singular past': 2,
               'past participle': 3, '3rd singular present': 4, 'infinitive': 5}
# --- --- #


def find_word_perturbation(sentence: str, label: str, target_idx: int) -> Tuple[str, str]:
    # Obtain pos tags and target token.
    doc = nlp(sentence)
    try:
        target_token = doc[target_idx]
    except:
        print(sentence, target_idx, doc)
    # Perturb input depending on the target token's pos tag.
    pos = target_token.pos_
    if pos == "NOUN":
        perturbation = perturb_noun(target_token)
    elif pos == "VERB":
        perturbation = perturb_verb(target_token)
    elif pos == "ADJ":
        perturbation, label = perturb_adjective(target_token, label)
    elif pos == "ADV":
        perturbation, label = perturb_adverb(target_token, label)
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
    # Return perturbed input and its label.
    return doc.text.replace(target_token.text, perturbation), label


def perturb_noun(token: spacy.tokens.token.Token) -> str:
    # Change singular nouns to plural nouns and vice versa.
    if token.tag_ == "NNS":
        # Plural noun, convert to singular.
        return nle.noun.singular(token.text)
    elif token.tag_ == "NN":
        # Singular noun, convert to plural.
        return nle.noun.plural(token.text)


def perturb_verb(token: spacy.tokens.token.Token) -> str:
    # Perturb verb by changing it to its present, progressive, past, perfect, 3rd person singular, or infinitive form.
    try:
        tense = VERB_TENSES[nle.verb.tense(token.text)]
    except KeyError:
        # Handle the case when the verb is unknown.
        return token.text
    form = np.random.choice([i for i in range(6) if i != tense])
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
    elif form == 5:
        # Infinitive.
        return nle.verb.infinitive(token.text)


def perturb_adjective(token: spacy.tokens.token.Token, label: str) -> Tuple[str, str]:
    text = replace_by_synonym(token)
    # Change label accordingly if token is changed for a synonym.
    label = label.replace(token.text, text)
    # Convert adjective to adverb. Ignoring irregular cases.
    if text.endswith("y"):
        return f"{text[:-1]}ily", label
    elif text.endswith("able") or text.endswith("ible") or text.endswith("le"):
        return f"{text[:-1]}y", label
    elif text.endswith("ic"):
        return f"{text}ally", label
    else:
        return f"{text}ly", label


def perturb_adverb(token: spacy.tokens.token.Token, label: str) -> Tuple[str, str]:
    text = replace_by_synonym(token)
    # Change label accordingly if token is changed for a synonym.
    label = label.replace(token.text, text)
    # Convert adverb to adjective. Ignoring irregular cases.
    if text.endswith("ily"):
        return f"{text[:-3]}y", label
    elif text.endswith("bly"):
        return f"{text[:-1]}e", label
    elif text.endswith("ically"):
        return text[:-4], label
    else:
        return text[:-2], label


def perturb_pronoun(token: spacy.tokens.token.Token) -> str:
    # Randomly pick a pronoun from a different pronoun type.
    pronoun = token.text.lower()
    if pronoun in SBJ_SET:
        pronoun_category = 0
        index = SBJ_PRONOUNS.index(pronoun)
    elif pronoun in OBJ_SET:
        pronoun_category = 1
        index = OBJ_PRONOUNS.index(pronoun)
    elif pronoun in POS_SET:
        pronoun_category = 2
        index = POS_PRONOUNS.index(pronoun)
    elif pronoun in REF_SET:
        pronoun_category = 3
        index = REF_PRONOUNS.index(pronoun)
    else:
        # Error handling.
        return token.text
    # Randomly pick one of the three complement categories.
    sample_category = np.random.randint(3)
    if np.random.uniform() < 0.1:
        # Randomly pick one of the forms in this category (in 10% of cases).
        return random.sample(COMPLEMENT_PRONOUNS[pronoun_category][sample_category], 1)[0]
    else:
        # Pick the corresponding pronoun from another category (in 90% of cases).
        return COMPLEMENT_PRONOUNS[pronoun_category][sample_category][index]


def perturb(token: spacy.tokens.token.Token) -> str:
    # Perturb word by removing token or replacing it with UNK symbol.
    if np.random.uniform() < 0.5:
        # Delete token.
        return ""
    else:
        return "<UNK>"


def replace_by_synonym(token: spacy.tokens.token.Token) -> str:
    # With 50% chance replace token by a random synonym.
    if np.random.uniform() < 0.5:
        return random.sample(get_synonyms(token), 1)[0]
    return token.text


def get_synonyms(token: spacy.tokens.token.Token) -> Set[str]:
    # Returns a set containing all synonyms over all synsets for the given token.
    synsets = wn.synsets(token.text, pos=spacy_tag_to_wordnet(token))
    synonyms = set(token.text)
    for sn in synsets:
        synonyms.update(sn.lemma_names())
    return synonyms


def spacy_tag_to_wordnet(token: spacy.tokens.token.Token) -> str:
    # Maps spacy tokens to POS tags used in wordnet.
    if token.tag_ in {"JJ", "JJR", "JJS"}:
        # Adjectives.
        return "a"
    elif token.tag_ in {"NN", "NNS"}:
        # Nouns.
        return "n"
    elif token.tag_ in {"RB", "RBR", "RBS", "WRB"}:
        # Adverbs.
        return "r"
    elif token.tag_ in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
        # Verbs.
        return "v"
    else:
        raise ValueError(f"Invalid POS tag: {token.tag_}")
