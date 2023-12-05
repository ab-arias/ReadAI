import nltk
from nltk.corpus import cmudict
import re
import syllables

nltk.download('punkt')
nltk.download('cmudict')

CURRENT_TAGS = ['<P>', '<S>', '</P>', '</S>']

# text = "The ineffable magniloquence of his erudition astounded the assemblage, as he expounded upon the labyrinthine intricacies of quantum entanglement with unparalleled perspicacity."
text = "I DO NOT LIKE THEM IN A HOUSE. I DO NOT LIKE THEM WITH A MOUSE. I DO NOT LIKE THEM HERE OR THERE. I DO NOT LIKE THEM ANYWHERE. I DO NOT LIKE GREEN EGGS AND HAM. I DO NOT LIKE THEM, SAM-I-AM."

para_text = text.split('\n\n')
meta_text = []
for para in para_text:
    sents = nltk.sent_tokenize(para)
    meta_sents = []
    for s in sents:
        words = nltk.word_tokenize(s)
        meta_sents.extend(['<S>'] + words + ['</S>'])

    meta_text.extend(['<P>'] + meta_sents + ['</P>'])

total_words = 0  # all processed words
total_syl = 0  # syls for processed words
total_sents = meta_text.count('<S>')


def flesch_grade_level(total_words, total_sents, total_syl):
    grade = 0.39*(total_words/total_sents)+11.8*(total_syl/total_words) - 15.59
    return grade

total_one_syl = 0  # new flesch
cmu_dict = cmudict.dict()
for word in meta_text:
    # if it doesn't start with 'and more letters or is just letters discard
    if not re.search(r'^\'[A-Za-z]+$|^[A-Za-z]+$', word):
        continue
    total_words += 1
    if word in cmu_dict.keys():  # check if it is in the dict
        curr_syls = 0
        for c in cmu_dict[word.lower()][0]:
            if c[len(c)-1].isnumeric():  # syllable markers have a number at the end (char 3)
                total_syl += 1
                curr_syls += 1
        if curr_syls == 1:
            total_one_syl += 1

    else:
        syls = syllables.estimate(word)
        total_syl += syls
        if syls == 1:
            total_one_syl += 1



print(
    f'Flesch Grade: {flesch_grade_level(total_words, total_sents, total_syl):.3f}')