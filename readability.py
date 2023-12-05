import sys
import os
import nltk
from nltk.corpus import cmudict
import re
import syllables
import math

nltk.download('punkt')
nltk.download('cmudict')

CURRENT_TAGS = ['<P>', '<S>', '</P>', '</S>']

# if len(sys.argv) != 2:
#     print(f'ERROR: Please provide ONE command line arg')
#     sys.exit(1)

# file_name = sys.argv[1]

# if not os.path.isfile(file_name):
#     print(f'ERROR: File DNE')
#     sys.exit(1)

text = "The ineffable magniloquence of his erudition astounded the assemblage, as he expounded upon the labyrinthine intricacies of quantum entanglement with unparalleled perspicacity."
# with open(f'./{file_name}', 'r') as f:
#     text = f.read()

para_text = text.split('\n\n')
meta_text = []
for para in para_text:
    sents = nltk.sent_tokenize(para)
    meta_sents = []
    for s in sents:
        words = nltk.word_tokenize(s)
        meta_sents.extend(['<S>'] + words + ['</S>'])

    meta_text.extend(['<P>'] + meta_sents + ['</P>'])


def flesch_reading_ease(total_one_syl, total_sents, total_words):
    score = 1.599*(100 * (total_one_syl / total_words)) - \
        1.015*(total_words/total_sents) - 31.517
    return score


def flesch_grade_level(total_words, total_sents, total_syl):
    grade = 0.39*(total_words/total_sents)+11.8*(total_syl/total_words) - 15.59
    return grade


# def dale_chall_read(difficult_words, total_words, total_sents):
#     pdw = difficult_words/total_words * 100
#     score = 0
#     if pdw < 5:
#         score = 0.1579*pdw + 0.0496*(total_words/total_sents)
#     else:
#         score = 0.1579*pdw + 0.0496*(total_words/total_sents) + 3.6365

#     grade = 0
#     if score < 5:
#         grade = '<=4'
#     elif score >= 5 and score < 6:
#         grade = '5–6'
#     elif score >= 6 and score < 7:
#         grade = '7–8'
#     elif score >= 7 and score < 8:
#         grade = '9–10'
#     elif score >= 8 and score < 9:
#         grade = '11-12'
#     elif score >= 9 and score < 10:
#         grade = '13-15'
#     elif score >= 10:
#         grade = '=>16'
#     return grade


def gunning_fog_index(total_words, total_sents, complex_words):
    index = 0.4*((total_words/total_sents) + 100 * (complex_words/total_words))
    return index


def SMOG_grade(total_poly_syl, total_sents):
    grade = 3+math.sqrt(30*(total_poly_syl/total_sents))
    return grade


# dale_chall_words = []
# with open('./texts/dale_chall.txt', 'r') as f:
#     dale_chall_words = f.read().split('\n')

total_sents = meta_text.count('<S>')
total_para = meta_text.count('<P>')

total_words = 0  # all processed words
total_syl = 0  # syls for processed words
difficult_words = 0
total_poly_syl = 0  # for SMOG and gunning fog (complex >= 3 as well)
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
        if curr_syls >= 3:  # Poly syl and gunning fog complex
            total_poly_syl += 1
    else:
        syls = syllables.estimate(word)
        total_syl += syls
        if syls == 1:
            total_one_syl += 1
        elif syls >= 3:
            total_poly_syl += 1
    # if word.lower() not in dale_chall_words:
    #     difficult_words += 1

# print(f'Total Sentences: {total_sents}')
# print(f'Total Paragraphs: {total_para}')
# print(f'Total Words: {total_words}')
# print(f'Total Syllables: {total_syl}')
# print(f'Total Polysyl: {total_poly_syl}')
# print(f'Difficult words: {difficult_words}')
# print(
#     f'Flesch: {flesch_reading_ease(total_one_syl, total_sents, total_words):.3f}')

# print(
#     f'dale: {dale_chall_read(difficult_words, total_words, total_sents)}')
# print(
#     f'gunning: {gunning_fog_index(total_words,total_sents, total_poly_syl):.3f}')

# print(f'SMOG: {SMOG_grade(total_poly_syl, total_sents):.3f}')


total_words = 0  # all processed words
total_sents = meta_text.count('<S>')
total_syl = 0  # syls for processed words




print(
    f'Flesch Grade: {flesch_grade_level(total_words, total_sents, total_syl):.3f}')