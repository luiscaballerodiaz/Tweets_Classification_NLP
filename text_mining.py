import pandas as pd
import numpy as np
from wordcloud import WordCloud
import extra_functions as f
import gensim.downloader as api
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from gridsearch_postprocess import GridSearchPostProcess


# MAIN
pd.set_option('display.max_columns', None)  # Enable option to display all dataframe columns
pd.set_option('display.max_rows', None)  # Enable option to display all dataframe rows
pd.set_option('display.max_colwidth', None)  # Enable printing the whole column content
pd.set_option('display.max_seq_items', None)  # Enable printing the whole sequence content

# Instantiate an object for GridSearchPostProcess to manage the grid search results
sweep = GridSearchPostProcess()

rng = np.random.RandomState(0)
param_sweep = False
count_en = True
tfidf_en = False
emb_en = False
count_tfidf_vocab = 2000
embedding = 'glove-twitter-100'

# Generate exercise dataframe
df = f.get_profner_data("./profner/")
print(df.iloc[890])
print(df.iloc[4492])
print(df.iloc[4770])
print(df.iloc[4945])
print(df.iloc[5019])
# Generate emoji dict from data --> https://www.clarin.si/repository/xmlui/handle/11356/1048
emoji_dict = f.load_emoji_sentiment("Emoji_Sentiment_Data_v1.0.csv")

# DATA OVERVIEW
print('Total tweets: {}\n'.format(df.shape[0]))
print('NA values:\n{}'.format(df.isna().sum()))
print('Total training tweets: {}\n'.format(df[df.set == 'train'].shape[0]))
print('Total testing tweets: {}\n'.format(df[df.set == 'test'].shape[0]))
print('Tweets duplicados: {}\n'.format(np.sum(df.duplicated(subset=['tweet_id']))))
print('Total set class distribution:\n{}\n'.format(df['label'].value_counts()))
print('Total set class distribution normalized:\n{}\n'.format(df['label'].value_counts(normalize=True)))
print('Training set class distribution:\n{}\n'.format(df.loc[df.set == 'train', 'label'].value_counts()))
print('Training set class distribution normalized:\n{}\n'.format(df.loc[df.set == 'train', 'label'].value_counts(
    normalize=True)))
print('Testing set class distribution:\n{}\n'.format(df.loc[df.set == 'test', 'label'].value_counts()))
print('Training set class distribution normalized:\n{}\n'.format(df.loc[df.set == 'train', 'label'].value_counts(
    normalize=True)))

df['tweet_length'] = df.apply(lambda row: len(row['tweet_text']), axis=1)
f.class_histogram(df, 'tweet_length', 'label', 'Histogram_Tweet_Length')

# TOKENIZATION
df['tweet_token'] = f.text_engineering(df['tweet_text'], http=True, hashtag=True, token=True, spaces=True, lower=True,
                                     punct=True, digits=True, stop=True, stem=True, mintoken=3, consecutive=True)
df['tweet_token_length'] = df.apply(lambda row: len(row['tweet_token']), axis=1)
f.class_histogram(df, 'tweet_token_length', 'label', 'Histogram_Tweet_Token_Length')

# WORDCLOUD
text_series = df['tweet_token'].apply(lambda row: ' '.join(row))
long_string = ' '.join(text_series.values.tolist())
wordcloud = WordCloud(background_color="white", max_words=count_tfidf_vocab, contour_width=0, contour_color='steelblue',
                      width=1200, height=600)
wordcloud.generate(long_string)
wordcloud.to_file('WordCloud.png')

# EMOJIS
df['emoji_list'] = df['tweet_text'].apply(lambda row: f.extract_emojis(row))
df['tweet_token'] = df.apply(lambda row: [x for x in row['tweet_token'] if x not in row['emoji_list']], axis=1)
df['sent_emoji_pos'] = df['emoji_list'].apply(lambda row: f.get_emoji_sentiment(row, 'positive', emoji_dict))
df['sent_emoji_neu'] = df['emoji_list'].apply(lambda row: f.get_emoji_sentiment(row, 'neutral', emoji_dict))
df['sent_emoji_neg'] = df['emoji_list'].apply(lambda row: f.get_emoji_sentiment(row, 'negative', emoji_dict))

# VECTORIZACION
count_vect = CountVectorizer(min_df=0.001, ngram_range=(1, 3), lowercase=False, max_features=count_tfidf_vocab)
tfidf_vect = TfidfVectorizer(norm=None, smooth_idf=False, min_df=0.001, ngram_range=(1, 3), lowercase=False,
                             max_features=count_tfidf_vocab)

count_data = count_vect.fit_transform(text_series).toarray()
tfidf_data = tfidf_vect.fit_transform(text_series).toarray()

bag_words = pd.DataFrame(count_data, columns=count_vect.get_feature_names_out())
bag_label = pd.concat([bag_words, df['label']], axis=1)
f.class_barword(bag_label, 25, 'Common_Words_per_Class')

print('Vocabulary size COUNT VECTORIZER: {}'.format(count_data.shape[1]))
print('Vocabulary size TFIDF VECTORIZER: {}'.format(tfidf_data.shape[1]))

# EMBEDDINGS
model_emb = api.load(embedding)
print('Vocabulary size EMBEDDING: {}'.format(model_emb.vectors.shape))
df['tweet_token_nostem'] = f.text_engineering(df['tweet_text'], http=True, hashtag=True, token=True, spaces=True,
                                            lower=True, punct=True, digits=True, stop=True, stem=False, mintoken=3)
df['tweet_token_emb'] = df['tweet_token_nostem'].apply(lambda row: np.mean(
    [model_emb.get_vector(x) for x in row if x in model_emb.vocab], axis=0))
df['tweet_token_emb'] = df['tweet_token_emb'].apply(lambda row: [0] * 100 if np.isnan(row).sum() == 1 else row)
emb_data = np.array(df['tweet_token_emb'].values.tolist())

extra_feats = df[['tweet_length', 'tweet_token_length', 'sent_emoji_pos', 'sent_emoji_neu', 'sent_emoji_neg']]
count = np.c_[(count_data, extra_feats)]
tfidf = np.c_[(tfidf_data, extra_feats)]
emb = np.c_[(emb_data, extra_feats)]

if param_sweep:
    param_grid = [{'preprocess': [StandardScaler()], 'estimator': [LogisticRegression()], 'estimator__n_jobs': [-1],
                   'estimator__penalty': ['l1', 'l2'], 'estimator__C': [0.1, 0.5, 1, 5, 10, 50, 100],
                   'estimator__solver': ['saga'], 'estimator__random_state': [rng]},
                  {'preprocess': [StandardScaler()], 'estimator': [LinearSVC()],
                   'estimator__C': [0.1, 0.5, 1, 5, 10, 50, 100], 'estimator__penalty': ['l1', 'l2'],
                   'estimator__random_state': [rng], 'estimator__dual': [False]},
                  {'preprocess': [None], 'estimator': [GaussianNB()]},
                  {'preprocess': [None], 'estimator': [LGBMClassifier()], 'estimator__n_estimators': [200],
                   'estimator__learning_rate': [0.1], 'estimator__num_leaves': [8, 14, 20, 30],
                   'estimator__max_depth': [3, 5, 8, 12], 'estimator__boosting_type': ['dart'],
                   'estimator__reg_alpha': [0.05], 'estimator__reg_lambda': [0.05], 'estimator__n_jobs': [-1],
                   'estimator__random_state': [rng]}]
else:
    param_grid = [{'preprocess': [StandardScaler()], 'estimator': [LogisticRegression()], 'estimator__n_jobs': [-1],
                   'estimator__penalty': ['l2'], 'estimator__C': [1], 'estimator__solver': ['saga'],
                   'estimator__random_state': [rng]}]
data_list = []
if count_en:
    data_list.append(count)
if tfidf_en:
    data_list.append(tfidf)
if emb_en:
    data_list.append(emb)
print('\nSIMULATION ON PROGRESS...')
for data in data_list:
    X_train = data[:5999]
    y_train = df['label'][:5999]
    X_test = data[6000:]
    y_test = df['label'][6000:]
    pipe = Pipeline([('preprocess', []), ('estimator', [])])
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=rng)  # Define the fold performance
    grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy')  # Define grid search cross validation
    grid_search.fit(X_train, y_train)  # Fit grid search for training set
    # Save a CSV file with the results for the grid search
    grid_results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
    print(grid_results)
    model = grid_search.best_estimator_  # Create model with best parametrization
    print("\nBEST MODEL PARAMETERS:\n{}".format(grid_search.best_params_))
    print("\nBEST MODEL CROSS VALIDATION SCORE: {:.4f}".format(grid_search.best_score_))
    print('\nBEST MODEL TEST SCORE: {:.4f}'.format(model.score(X_test, y_test)))

sweep.param_sweep_matrix(params=grid_results['params'], test_score=grid_results['mean_test_score'])
if count_en and not tfidf_en and not emb_en and not param_sweep:
    coef_list = model['estimator'].coef_
    word_list = list(count_vect.get_feature_names_out())
    word_list.extend(['tweet_length', 'tweet_token_length', 'sent_emoji_pos', 'sent_emoji_neu', 'sent_emoji_neg'])
    f.bar_coefficient(coef_list, word_list, 25, 'Coefficient Significance')
