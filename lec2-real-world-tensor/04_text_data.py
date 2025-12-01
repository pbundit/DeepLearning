import torch


def load_text_data():
    with open('data/text_data/1342-0.txt', encoding='utf8') as f:
        text = f.read()
    lines = text.split('\n')

    return lines


def clean_words(input_str): #ทำให้dictเราสะอาดและเล้กลง 
    punctuation = '.,;:"!?`_-()\'[]*'
    word_list = input_str.lower().replace('\n', ' ').split()
    word_list = [word.strip(punctuation)
                 for word in word_list]  # remove all the punctuation
    return word_list


def create_dictionary(word_list): 
    words_in_data = sorted(set(word_list))  # remove duplicate
    return {word: i for (i, word) in enumerate(words_in_data)} #ทุก element ใส่เลขไป


def create_sentence_tensor(sentence, dictionary):
    words_in_sentence = clean_words(sentence)
    dictionary_size = 8394
    sentence_tensor = torch.zeros(len(words_in_sentence), dictionary_size)
    for i, word in enumerate(words_in_sentence):
        word_index = dictionary[word]
        sentence_tensor[i][word_index] = 1

    return sentence_tensor


if __name__ == "__main__":
    lines = load_text_data()
    #print(lines[: 5])
    print(len(lines))
    dictionary = create_dictionary(clean_words(lines[100]))
    print(dictionary)
    one_line_sentence = lines[200]
    print(one_line_sentence)
