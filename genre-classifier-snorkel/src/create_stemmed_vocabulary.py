from nltk.stem import PorterStemmer
import json

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

if '__name__' == '__main__':
    for json_file in ['0.json', '1.json', '2.json', '3.json', '4.json', '5.json', '6.json', '7.json', 'all_terms.json']:
        with open(directory + '/Json/' + json_file) as data:
            dict_data = json.load(data)
            stem_to_sum = stemmed_word_dictonary(dict_data)
        with open(directory + '/Json_stemmed_word/' + json_file, "w") as outputfile:
            json.dump(stem_to_sum, outputfile)
    dict_tf = {}
    for json_file in ['0.json', '1.json', '2.json', '3.json', '4.json', '5.json', '6.json', '7.json']:
        with open(directory + '/Json_stemmed_word/' + json_file) as data:
            dict_tf[int(json_file[0])] = json.load(data)
    with open(directory + '/Json_stemmed_word/all_terms.json') as data:
        dict_df = json.load(data)

    dict_tf_idf = {}

    for key, value in dict_tf.items():
        # Initialize dict_tf_idf[key] if it doesn't exist
        if key not in dict_tf_idf:
            dict_tf_idf[key] = {}
        
        for word, tf in value.items():
            if word in dict_df:
                df = dict_df[word]
                dict_tf_idf[key][word] = tf / df

    for key, value in dict_tf_idf.items():
        # Sort the dictionary by value in descending order and take only the highest 100
        sorted_value = dict(sorted(value.items(), key=lambda item: item[1], reverse=True)[:200])
        dict_tf_idf[key] = sorted_value

    dict_tf_idf_set = {}
    for key, value in dict_tf_idf.items():
        dict_tf_idf_set[key] = set([word for word, _ in value.items()]) 

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
        with open(directory + '/vocabulary_modified/vocabulary-popsecul-modified-' + str(key) + ".txt", 'w') as f:
            f.write(' '.join(value))