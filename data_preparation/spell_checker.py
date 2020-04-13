from collections import Counter
from sklearn.datasets import fetch_20newsgroups
import re

'''
def get_symspell_object():
    corpus = []

    for line in fetch_20newsgroups().data:
        line = line.replace('\n', ' ').replace('\t', ' ').lower()
        line = re.sub('[^a-z ]', ' ', line)
        tokens = line.split(' ')
        tokens = [token for token in tokens if len(token) > 0]
        corpus.extend(tokens)

    corpus = Counter(corpus)
    corpus_dir = '../NOTEEVENTS.csv'
    corpus_file_name = 'spell_check_dictionary.txt'
    symspell = SymSpell(verbose=10)
    symspell.build_vocab(
        dictionary=corpus,
        file_dir=corpus_dir, file_name=corpus_file_name)
    symspell.load_vocab(corpus_file_path=corpus_dir+corpus_file_name)

    return symspell


get_symspell_object()
'''

'''
from spellchecker import SpellChecker


def get_correctly_spelled(text, spell):
    #text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()

    # find those words that may be misspelled
    misspelled = spell.unknown(text)
    print(type(text))

    for i in range(len(text)):
        # Get the one `most likely` answer
        text[i] = text[i].replace(' ', '')
        # print('start ', text[i], ' end')
        correct = spell.correction(text[i])
        if text[i] != correct:
            print(text)
            text[i] = correct
            print(text)

    return ' '.join(text)
'''
