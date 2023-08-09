import os
import string
import re
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from emoji_extractor.extract import Extractor


def get_tweet_content(list_paths):
    """Función para guardar en un diccionario el contenido de archivos txt que se introduce en su entrada.
    Devuelve un diccionario en el que las claves son el id del tweet, y el valor el texto del tweet."""
    output_dict = dict()
    for i in list_paths:
        tweet_id = i.split("/")[-1].split(".txt")[0]
        with open(i, 'r', encoding='utf8') as f:
            output_dict[int(tweet_id)] = f.read()
    return output_dict


def get_profner_data(profner_path_data):
    # Obtenemos el path a los txt de los tweets.
    path_to_txt = profner_path_data+"subtask-1/train-valid-txt-files/"
    tweets_train_files = [path_to_txt+"train/"+i for i in os.listdir(path_to_txt+"train/")]
    tweets_valid_files = [path_to_txt+"valid/"+i for i in os.listdir(path_to_txt+"valid/")]
    # Obtenemos diccionarios en los que el key es el tweet_id y el value el texto del tweet.
    train_txt_content = get_tweet_content(tweets_train_files)
    valid_txt_content = get_tweet_content(tweets_valid_files)

    # Cargamos dos dataframes con los tweet_id y la categoría de los tweets
    path_to_labeled = profner_path_data+"subtask-1/"
    train_tweets = pd.read_csv(path_to_labeled+"train.tsv",sep="\t")
    valid_tweets = pd.read_csv(path_to_labeled+"valid.tsv",sep="\t")

    # Introducimos a los df el campo de texto mapeando los diccionarios con tweet_id
    train_tweets["tweet_text"] = train_tweets['tweet_id'].map(train_txt_content)
    train_tweets["set"] = "train"
    valid_tweets["tweet_text"] = valid_tweets['tweet_id'].map(valid_txt_content)
    valid_tweets["set"] = "test"

    # Concatenamos el resultado
    output_df = pd.concat([train_tweets,valid_tweets],axis=0)
    # Eliminamos retorno de carro
    output_df["tweet_text"] = output_df.tweet_text.apply(lambda x: x.replace('\n', ' '))
    return output_df[["tweet_id","tweet_text","label","set"]].reset_index(drop=True)


def load_emoji_sentiment(path):
    # Cargamos el csv de emoji_sentiment
    emoji_sent_df = pd.read_csv(path, sep=",")
    # Calculamos los scores dividiendo el número de emojis de cada sentimiento y entre el total
    emoji_sent_df["Negative"] = emoji_sent_df["Negative"] / emoji_sent_df["Occurrences"]
    emoji_sent_df["Neutral"] = emoji_sent_df["Neutral"] / emoji_sent_df["Occurrences"]
    emoji_sent_df["Positive"] = emoji_sent_df["Positive"] / emoji_sent_df["Occurrences"]
    # Transformamos a dict
    emoji_sent_df = emoji_sent_df.set_index('Emoji')
    emoji_dict = emoji_sent_df.to_dict(orient="index")
    return emoji_dict


def extract_emojis(text):
    extract = Extractor()
    emojis = extract.count_emoji(text, check_first=False)
    emojis_list = [key for key, _ in emojis.most_common()]
    return emojis_list


def get_emoji_sentiment(lista, option, emoji_sent_dict):
    output = 0
    for emoji in lista:
        try:
            if option == "positive":
                output = output + emoji_sent_dict[emoji]["Positive"]
            elif option == "negative":
                output = output + emoji_sent_dict[emoji]["Negative"]
            elif option == "neutral":
                output = output + emoji_sent_dict[emoji]["Neutral"]
        except (Exception,):
            continue
    return output


def class_histogram(df1, feat, target, name='Histogram'):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    ax = axes.ravel()
    bins_vector = list(np.linspace(np.min(df1[feat]) - 1E-3, np.max(df1[feat]) + 1E-3, 20))
    norm = [False, True]
    for ind in range(2):
        ax[ind].hist(df1.loc[df1[target] == 0, feat], bins=bins_vector, density=norm[ind], color='b', alpha=0.5,
                     label='0')
        ax[ind].hist(df1.loc[df1[target] == 1, feat], bins=bins_vector, density=norm[ind], color='r', alpha=0.5,
                     label='1')
        ax[ind].grid(visible=True)
        ax[ind].legend()
        ax[ind].tick_params(axis='both', labelsize=10)
        ax[ind].set_xlabel('Length', fontsize=10, weight='bold')
    ax[0].set_title('Histogram', fontsize=20, weight='bold')
    ax[1].set_title('Normalized histogram', fontsize=20, weight='bold')
    ax[0].set_ylabel('Frequency', fontsize=10, weight='bold')
    ax[1].set_ylabel('Normalized Frequency', fontsize=10, weight='bold')
    fig.tight_layout()
    plt.savefig(name + '.png', bbox_inches='tight')
    plt.close()


def class_barword(bag, max_words=25, name='Barword'):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    ax = axes.ravel()

    bag_0 = bag.loc[bag['label'] == 0]
    bag_0.drop('label', axis=1, inplace=True)
    bag_0_sum = np.sum(bag_0, axis=0)
    ind_0 = list(np.argsort(-bag_0_sum))
    bars = [bag_0_sum[ind_0[x]] for x in range(max_words)]
    names = [bag_0.columns.values.tolist()[ind_0[x]] for x in range(max_words)]
    ax[0].barh(range(1, max_words + 1), bars, color='b', height=0.25, edgecolor='black')
    ax[0].set_yticks(range(1, max_words + 1), names, rotation=0)

    bag_1 = bag.loc[bag['label'] == 1]
    bag_1.drop('label', axis=1, inplace=True)
    bag_1_sum = np.sum(bag_1, axis=0)
    ind_1 = list(np.argsort(-bag_1_sum))
    bars = [bag_1_sum[ind_1[x]] for x in range(max_words)]
    names = [bag_1.columns.values.tolist()[ind_1[x]] for x in range(max_words)]
    ax[1].barh(range(1, max_words + 1), bars, color='r', height=0.25, edgecolor='black')
    ax[1].set_yticks(range(1, max_words + 1), names, rotation=0)

    for ind in range(2):
        ax[ind].grid(visible=True)
        ax[ind].tick_params(axis='x', labelsize=10)
        ax[ind].set_xlabel('Occurrence', fontsize=10, weight='bold')
        ax[ind].set_ylabel('Common words', fontsize=10, weight='bold')
    ax[0].set_title('Most common words for class 0', fontsize=20, weight='bold')
    ax[1].set_title('Most common words for class 1', fontsize=20, weight='bold')
    fig.tight_layout()
    plt.savefig(name + '.png', bbox_inches='tight')
    plt.close()


def bar_coefficient(coefs_ini, words, max_coefs=25, name='Bar Coefficient'):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    ax = axes.ravel()
    coefs = coefs_ini.ravel()
    coef_ind_max = np.argsort(-coefs)
    coef_ind_min = np.argsort(coefs)
    word_max = [words[coef_ind_max[x]] for x in range(max_coefs)]
    word_min = [words[coef_ind_min[x]] for x in range(max_coefs)]
    coef_max = [coefs[coef_ind_max[x]] for x in range(max_coefs)]
    coef_min = [coefs[coef_ind_min[x]] for x in range(max_coefs)]
    ax[0].barh(range(1, max_coefs + 1), coef_max, color='b', height=0.25, edgecolor='black')
    ax[0].set_yticks(range(1, max_coefs + 1), word_max, rotation=0)
    ax[1].barh(range(1, max_coefs + 1), coef_min, color='r', height=0.25, edgecolor='black')
    ax[1].set_yticks(range(1, max_coefs + 1), word_min, rotation=0)
    for ind in range(2):
        ax[ind].grid(visible=True)
        ax[ind].tick_params(axis='x', labelsize=10)
        ax[ind].set_xlabel('Coefficient magnitude', fontsize=10, weight='bold')
        ax[ind].set_ylabel('Words', fontsize=10, weight='bold')
    ax[0].set_title('Most significant words', fontsize=20, weight='bold')
    ax[1].set_title('Least significant words', fontsize=20, weight='bold')
    fig.tight_layout()
    plt.savefig(name + '.png', bbox_inches='tight')
    plt.close()


def text_engineering(df_old, http=True, hashtag=True, token=True, spaces=True, lower=True, punct=True, digits=True,
                     stop=True, stem=True, mintoken=1, consecutive=True):
    stemmer = SnowballStemmer('spanish')
    stop_words = set(stopwords.words('spanish'))
    punctuations = list(string.punctuation)
    punctuations.extend(['„', '“', '”', '¿', '€', '$'])
    out = df_old.copy()
    out = out.apply(lambda row: row.replace('ñ', 'n'))
    out = out.apply(lambda row: row.replace('í', 'i'))
    out = out.apply(lambda row: row.replace('ó', 'o'))
    out = out.apply(lambda row: row.replace('ú', 'u'))
    out = out.apply(lambda row: row.replace('é', 'e'))
    out = out.apply(lambda row: row.replace('á', 'a'))
    if http:  # Remove https websites
        out = out.apply(lambda row: re.sub(r'(\shttps\S+)', ' ', row))
    if hashtag:  # Separate with whitespaces non spaces words in a chain
        out = out.apply(lambda row: ' '.join(re.split(r'([A-Z][a-z]+)', row)))
    if lower:  # Transform to lowercase
        out = out.apply(lambda row: row.lower())
    if consecutive:  # Eliminate consecutive chars
        out = out.apply(lambda row: re.sub(r'((\w)\2{3,})', r'\2', row))
    if token:  # Tokenization with NLTK
        out = out.apply(lambda row: word_tokenize(row))
    if spaces:  # Remove extra spaces if needed
        out = out.apply(lambda row: [x.strip() for x in row])
    if punct:  # Remove tokens with punctuation marks and remove punctuation marks chars from string tokens
        for char in punctuations:
            out = out.apply(lambda row: [x.replace(char, '') for x in row])
        out = out.apply(lambda row: [x for x in row if x != ''])
    if digits:  # Remove tokens with all chars digits
        out = out.apply(lambda row: [x for x in row if not x.isdigit()])
    if stop:  # Remove stopwords with NLTK
        out = out.apply(lambda row: [x for x in row if x not in stop_words])
    if stem:  # Snowball Stemmer with NLTK
        out = out.apply(lambda row: [stemmer.stem(x) for x in row])
    out = out.apply(lambda row: [x for x in row if len(x) >= mintoken])
    return out
