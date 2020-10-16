from collections import defaultdict
import re
import requests
from sklearn.feature_extraction import text
from newspaper import Article
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_stop_words(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

def parse(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    return article

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

def tfidf(article):
    tf_idf_vector=tfidf_transformer.transform(cv.transform([article]))
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    return keywords


#load a set of stop words
stopwords=get_stop_words("stopwords_en.txt")

urls = {'gdp': {'r':'https://www.breitbart.com/economy/2019/07/26/u-s-economy-grew-at-2-1-pace-in-second-quarter/',
                 'l':'https://www.vox.com/2019/7/26/8931569/gdp-q2-trump-tax-failed'},
        'death': {'l':'https://www.motherjones.com/politics/2019/07/a-federal-judge-just-blocked-trumps-latest-attempt-to-turn-away-asylum-seekers/',
                    'r':'https://dailycaller.com/2019/07/25/ag-barr-reinstates-federal-death-penalty/'},
        'rally': {'r': 'https://www.breitbart.com/politics/2019/07/17/watch-trump-rally-chants-send-her-back-after-president-slams-ilhan-omar/',
                  'l':'https://www.huffpost.com/entry/send-her-back-trump-racist-chant-ilhan-omar_n_5d2fb9bce4b0419fd327c264'},
        'mueller':{'l':'https://www.nbcnews.com/politics/justice-department/robert-mueller-make-public-statement-about-russia-probe-wednesday-n1011331',
                   'r':'https://www.foxnews.com/politics/barr-says-he-didnt-agree-with-the-legal-analysis-in-mueller-report-says-it-did-not-reflect-the-views-of-doj'}}

results2 = defaultdict(dict)

for topic in urls:
    both_sides = urls[topic]
    nk_left = parse(both_sides['l'])
    nk_right = parse(both_sides['r'])

    cv=text.CountVectorizer(ngram_range=(1,2), max_df=1,stop_words='english')
    word_count_vector=cv.fit_transform([nk_left.text, nk_right.text])
    feature_names=cv.get_feature_names()

    tfidf_transformer=text.TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    # get the document that we want to extract keywords from

     #generate tf-idf for the given document
    nk_left_kw = tfidf(nk_left.text)
    nk_right_kw = tfidf(nk_right.text)

    results2[topic]['left'] = [(k,nk_left_kw[k]) for k in nk_left_kw]
    results2[topic]['right'] = [(k,nk_right_kw[k]) for k in nk_right_kw]
    results2[topic]['both'] = list(set(nk_right.keywords).intersection(set(nk_left.keywords)))

    # now print the results
    # print('UNIQUE WORDS')
    # print("\n=====Left=====")
    # for k in nk_left_kw:
    #     print(k,nk_left_kw[k])
    # print("\n=====Right=====")
    # for k in nk_right_kw:
    #     print(k,nk_right_kw[k])
    #
    # print("\n=====OVERALL KEYWORDS=====")
    # for kw in set(nk_right.keywords).intersection(set(nk_left.keywords)):
    #     print(kw)

for topic in results:
    print('=============================================')
    print(topic.upper(), results[topic]['both'])
    print('=============================================')
    print('-----------------LEFT------------------------')
    print(results[topic]['left'])
    print('----------------RIGHT------------------------')
    print(results[topic]['right'])
    print()

for topic in results2:
    print('=============================================')
    print(topic.upper(), results2[topic]['both'])
    print('=============================================')
    print('-----------------LEFT------------------------')
    print(results2[topic]['left'])
    print('----------------RIGHT------------------------')
    print(results2[topic]['right'])
    print()


## PLOTTING WORD CLOUD
mask = np.array(Image.open(requests.get("https://previews.123rf.com/images/alancotton/alancotton1507/alancotton150700344/42853419-an-outline-silhouette-map-of-the-united-states-of-america-over-a-white-background.jpg", stream=True).raw))

urls
left_article = parse(urls['rally']['l'])
right_article = parse(urls['rally']['r'])

red = ' '.join([x[0] for x in results2['rally']['right']])
blue = ' '.join([x[0] for x in results2['rally']['left']])
neutral = ' '.join(results2['rally']['both'])

def left_or_right(word, *args, **kwargs):
    if word in red:
        return '#9e1010'
    elif word in blue:
        return '#1d48a3'
    elif word in left_article.text:
        return '#6baed6'
    elif word in right_article.text:
        return '#fcbba1'
    else:
        return '#4f4c4c'

def generate_wordcloud(words, mask):
    word_cloud = WordCloud(width = 2048, height = 1536, color_func=left_or_right, background_color='white', stopwords=stopwords, mask=mask).generate(words)
    plt.figure(figsize=(16,10),facecolor = 'white', edgecolor='blue')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('wordcloud_1.png')
    plt.show()



fig1 = generate_wordcloud(f'{red} {blue} {red} {blue} {red} {blue} {neutral} {right_article.text} {left_article.text}', mask)
