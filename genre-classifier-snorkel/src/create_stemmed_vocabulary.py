from nltk.stem import PorterStemmer
import json
from genre_classification_rules import label_names

# this file is a preprocessing file to improve the word lists for the classes
# the idea is to consider each class as document and calculate the tf_idf for each class
# to get set of unique words in each class, that can define the class

directory = "/workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/resources"

def stemmed_word_dictonary(dict):
    stemmer = PorterStemmer()
    stem_to_sum = {}

    for word, number in dict.items():
        stemmed_word = stemmer.stem(word).lower()
        if stemmed_word in stem_to_sum:
            stem_to_sum[stemmed_word] += number
        else:
            stem_to_sum[stemmed_word] = number
    return stem_to_sum

def stem_word_lists():
    # stem the words 
    for json_file in ['0.json', '1.json', '2.json', '3.json', '4.json', '5.json', '6.json', '7.json', 'all_terms.json']:
        with open(directory + '/Json/' + json_file) as data:
            dict_data = json.load(data)
            stem_to_sum = stemmed_word_dictonary(dict_data)
        # save the stemmed words in json files
        with open(directory + '/Json_stemmed_word/' + json_file, "w") as outputfile:
            json.dump(stem_to_sum, outputfile)

def get_tf_df():
    # calc the term frequency for each class
    dict_tf = {}
    for json_file in ['0.json', '1.json', '2.json', '3.json', '4.json', '5.json', '6.json', '7.json']:
        with open(directory + '/Json_stemmed_word/' + json_file) as data:
            dict_tf[int(json_file[0])] = json.load(data)
    with open(directory + '/Json_stemmed_word/all_terms.json') as data:
        dict_df = json.load(data)
    return dict_tf, dict_df

def calc_tf_idf(dict_tf, dict_df):
    dict_tf_idf = {}
    for key, value in dict_tf.items():
        # Initialize dict_tf_idf[key] if it doesn't exist
        if key not in dict_tf_idf:
            dict_tf_idf[key] = {}
        
        for word, tf in value.items():
            if word in dict_df:
                df = dict_df[word]
                dict_tf_idf[key][word] = tf / df
    return dict_tf_idf

def get_key_words(dict_tf_idf, num=100):
    for key, value in dict_tf_idf.items():
        # Sort the dictionary by value in descending order and take only the highest 100
        sorted_value = dict(sorted(value.items(), key=lambda item: item[1], reverse=True)[:num])
        dict_tf_idf[key] = sorted_value

    dict_tf_idf_set = {}
    for key, value in dict_tf_idf.items():
        dict_tf_idf_set[key] = set([word for word, _ in value.items()])
    return dict_tf_idf_set

def make_disjoint(dict_tf_idf_set):
    elements_count = {}
    disjoint_dict = {}
    for key, value in dict_tf_idf_set.items():
        for element in value:
            if element in elements_count:
                elements_count[element] += 1
            else:
                elements_count[element] = 1

    for key, value in dict_tf_idf_set.items():
        new_set = set()
        for element in value:
            if elements_count[element] == 1:
                new_set.add(element)
        disjoint_dict[key] = new_set
    for key, value in disjoint_dict.items():
        with open(directory + '/vocabulary-modified/vocabulary-popsecul-modified-' + label_names[key].lower().replace(" ", "-") + ".txt", 'w') as f:
            f.write(' '.join(value))

if __name__ == '__main__':
     dict_tf, dict_df = get_tf_df()
     res = calc_tf_idf(dict_tf, dict_df)
     words_set = get_key_words(res, num=50)
     make_disjoint(words_set)
