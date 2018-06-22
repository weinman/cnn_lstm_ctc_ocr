def get_words():
    with open('../data/lexicon.txt', 'r') as words_file:
        return [line[:-1] for line in words_file]
    

