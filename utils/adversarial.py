import spacy
import random
import numpy as np
from typing import Tuple, Set, List
# Docs: https://www.nodebox.net/code/index.php/Linguistics.html. Install via requirements.txt.
import nodebox_linguistics_extended as nle
from nltk.corpus import wordnet as wn


# Set random seed to system time; script will be called multiple times and perturbations should differ.
random.seed()
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
# Prepositions #
PREP = ["with", "at", "to", "from", "into", "against", "of", "in", "for", "on", "by", "about", "like", "over", "after",
        "since", "off"]
# --- --- #
# Auxiliaries #
AUX = ["be", "can", "could", "dare", "do", "have", "may", "might", "must", "need", "ought", "shall", "should",
       "will", "would"]
AUX_REPLACEMENT = {"be", "have"}


# --- --- #
# Helper functions #


def find(doc: spacy.tokens.Doc, idx: int) -> int:
    # Find how many times the token specified by idx occurs before idx.
    word = doc[idx]
    counter = 0
    for i, token in enumerate(doc):
        if token.text == word.text and i != idx:
            counter += 1
    return counter


def convert_sentence(sent: str, label: str, model, num_perturbations: int) -> Tuple[List[Tuple], np.ndarray, List[str],
                                                                                    List[int]]:
    # Get the aggregated attention weights for each token, generate adversarial examples.
    tokens = [token.text for token in nlp(sent)]
    weights = model.extract_candidate_words([tokens], return_attention=True)
    weights /= np.sum(weights)
    perturbations, indices = [], []
    for i in range(num_perturbations):
        perturbation, label_temp, _, indices_temp = random_perturbation(sent, label)
        perturbations.append((perturbation, label_temp))
        indices.extend(indices_temp)
    return perturbations, weights, tokens, indices

# --- --- #


def random_perturbation(sentence: str, label: str, num: int = 1) -> Tuple[
        List[str], List[str], List[str], List[int]]:
    # Choose a random index w.r.t. the spacy tokenisation.
    sent_length = len(nlp(sentence))
    idx = random.sample(range(0, sent_length), np.minimum(num, sent_length))
    # Perturb sentence according to rules.
    return find_word_perturbation(sentence, label, idx, len(idx))


def find_word_perturbation(sentence: str, label: str, target_idx: List[int], num: int = 1) -> Tuple[
        List[str], List[str], List[str], List[int]]:
    perturbations, labels, pos_list, indices = [], [], [], []
    for i in range(num):
        # Obtain pos tags and target token.
        doc = nlp(sentence)
        # Handle empty sentences.
        if len(doc) <= target_idx[i]:
            continue
        target_token = doc[target_idx[i]]
        occurrence = find(doc, target_idx[i])
        # Perturb input depending on the target token's pos tag.
        pos = target_token.pos_
        if pos == "NOUN":
            perturbation = perturb_noun(target_token)
        elif pos == "AUX":
            perturbation = perturb_auxiliary(target_token)
        elif pos == "VERB":
            perturbation = perturb_verb(target_token.text)
        elif pos == "ADJ":
            perturbation, label = perturb_adjective(target_token, occurrence, label)
        elif pos == "ADV":
            perturbation, label = perturb_adverb(target_token, occurrence, label)
        elif pos == "PRON":
            perturbation = perturb_pronoun(target_token)
        elif pos == "ADP":
            perturbation = perturb_preposition(target_token)
        elif pos == "NUM" or pos == "PROPN" or pos == "PART":
            # Remain unchanged.
            perturbation = target_token.text
            # For analysing which rules were used.
            pos = "NUM-PROPN-PART"
        elif pos == "PUNCT":
            # Delete punctuation.
            perturbation = ""
        else:
            # Delete or UNK.
            perturbation = perturb(target_token)
            pos = "OTHER"
        # Return perturbed input and its label.
        sentence = generate_output(doc, perturbation, target_idx[i])
        perturbations.append(sentence)
        labels.append(label)
        pos_list.append(pos)
        if len(perturbation) > 0:
            indices.append(target_idx[i])
        else:
            indices.append(-1)
    return perturbations, labels, pos_list, indices


def generate_output(doc: spacy.tokens.Doc, perturbation: str, idx: int) -> str:
    # Replace token in input with perturbation.
    if len(doc) > idx + 1:
        if idx == 0:
            return f"{perturbation} {doc[idx + 1:]}" if perturbation != "" else doc[idx + 1:].text
        else:
            return f"{doc[:idx]} {perturbation} {doc[idx + 1:]}" if perturbation != "" else f"{doc[:idx]} {doc[idx + 1:]}"
    else:
        if idx == 0:
            return f"{perturbation}" if perturbation != "" else ""
        else:
            return f"{doc[:idx]} {perturbation}" if perturbation != "" else doc[:idx].text


def generate_label(label: str, token: str, replacement: str, occurrence: int) -> str:
    return label.replace(token, "<$$$>", occurrence).replace(token, replacement).replace("<$$$>", token)


def perturb_auxiliary(token: spacy.tokens.token.Token) -> str:
    inf = nle.verb.infinitive(token.text)
    if inf not in AUX_REPLACEMENT:
        word = random.sample(AUX_REPLACEMENT, 1)[0]
        return perturb_verb(word)
    else:
        if inf == "be":
            return perturb_verb("have")
        else:
            return perturb_verb("be")


def perturb_noun(token: spacy.tokens.token.Token) -> str:
    # Change singular nouns to plural nouns and vice versa.
    if token.tag_ == "NNS":
        # Plural noun, convert to singular.
        return nle.noun.singular(token.text)
    elif token.tag_ == "NN":
        # Singular noun, convert to plural.
        return nle.noun.plural(token.text)


def perturb_verb(token: str) -> str:
    # Perturb verb by changing it to its present, progressive, past, perfect, 3rd person singular, or infinitive form.
    try:
        token_tense = nle.verb.tense(token)
        tense = VERB_TENSES[token_tense]
        negated = nle.verb.is_tense(token, token_tense, negated=True)
    except KeyError:
        # Handle the case when the verb is unknown.
        return token
    # Only consider appropriate tenses.
    tenses = [i for i in [0, 2, 4] if i != tense] if negated else [i for i in range(6) if i != tense]
    perturbation = token
    # Loop multiple times in case NLE returns an empty string. If no valid perturbation is found the input is returned.
    for i in range(10):
        form = np.random.choice(tenses)
        if form == 0:
            # Present form.
            perturbation = nle.verb.present(token, negate=negated)
        elif form == 1:
            # Progressive form.
            perturbation = nle.verb.present_participle(token)
        elif form == 2:
            # Past form.
            perturbation = nle.verb.past(token, negate=negated)
        elif form == 3:
            # Perfect form.
            perturbation = nle.verb.past_participle(token)
        elif form == 4:
            # 3rd person singular form.
            perturbation = nle.verb.present(token, person=3, negate=negated)
        elif form == 5:
            # Infinitive.
            perturbation = nle.verb.infinitive(token)
        # NLE can return an empty string without throwing an error. There is no easy way to avoid this.
        if perturbation != '' and perturbation != token:
            break
    return perturbation


def perturb_adjective(token: spacy.tokens.token.Token, occurrence: int, label: str) -> Tuple[str, str]:
    text = replace_by_synonym(token)
    # Change label accordingly if token is changed for a synonym.
    label = generate_label(label, token.text, text, occurrence)
    # Convert adjective to adverb. Ignoring irregular cases.
    if text.endswith("y"):
        return f"{text[:-1]}ily", label
    elif text.endswith("able") or text.endswith("ible") or text.endswith("le"):
        return f"{text[:-1]}y", label
    elif text.endswith("ic"):
        return f"{text}ally", label
    else:
        return f"{text}ly", label


def perturb_adverb(token: spacy.tokens.token.Token, occurrence: int, label: str) -> Tuple[str, str]:
    text = replace_by_synonym(token)
    # Change label accordingly if token is changed for a synonym.
    label = generate_label(label, token.text, text, occurrence)
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


def perturb_preposition(token: spacy.tokens.token.Token) -> str:
    return random.sample(PREP, 1)[0]


def perturb(token: spacy.tokens.token.Token) -> str:
    # Delete token.
    return ""


def replace_by_synonym(token: spacy.tokens.token.Token) -> str:
    # With 50% chance replace token by a random synonym.
    if np.random.uniform() < 0.5:
        return random.sample(get_synonyms(token), 1)[0]
    return token.text


def get_synonyms(token: spacy.tokens.token.Token) -> Set[str]:
    # Returns a set containing all synonyms over all synsets for the given token.
    wn_tag = spacy_tag_to_wordnet(token)
    if wn_tag == "":
        return token.text
    synsets = wn.synsets(token.text, pos=wn_tag)
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
        return ""
