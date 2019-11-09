from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk import bigrams
from collections import Counter


def frequency_distribution(text):
    words = word_tokenize(text)
    freq_dist = FreqDist(words)

    print(freq_dist.most_common(15))


def longer_words_appear_the_most(text, min_length, min_frequency):

    words = word_tokenize(text)
    freq_dist = FreqDist(words)
    print(sorted(
         [w for w in set(words)
          if len(w) >= min_length and freq_dist[w] > min_frequency]
    ))


def get_bigrams(text):
    words = word_tokenize(text)
    extracted_bigrams = bigrams(text.split(" "))

    bigrams2 = zip(words, words[1:])
    counts = Counter(bigrams2)
    print(counts.most_common()[:50])
