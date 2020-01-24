from nltk import word_tokenize
import string


def get_clean_text(data):
    # nan filled in with an empty space
    data.TEXT = data.TEXT.fillna(' ')
    data.TEXT = data.TEXT.str.replace('\n',' ')
    data.TEXT = data.TEXT.str.replace('\r',' ')
    return data


def get_tokenizer(text):

    t = str.maketrans(dict.fromkeys(string.punctuation + '0123456789', " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens


def get_stop_words():

    my_stop_words = ['the','and','to','of','was','with','a','on','in','for','name',
                 'is','patient','s','he','at','as','or','one','she','his','her','am',
                 'were','you','pt','pm','by','be','had','your','this','date',
                 'from','there','an','that','p','are','have','has','h','but','o',
                 'namepattern','which','every','also', 'should', 'if', 'it', 'been',
                 'who', 'during', 'any', 'c', 'd', 'x']
    return my_stop_words



