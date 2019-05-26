from django.shortcuts import render
from django.http import HttpResponse
import firebase_admin
import pandas as pd 
import numpy as np
import os
import gmaps #API key AIzaSyBzhOePBXX2opJlMj41yo1pfDujrmTEPi0
import gmaps.datasets
from ipywidgets.embed import embed_minimal_html
from firebase import firebase
from django.conf import settings
from firebase_admin import credentials
from firebase_admin import firestore
#---imports for hamza code----#
import re
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import re
from nltk.corpus import state_union, stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import plotly
from matplotlib.backends.backend_agg import FigureCanvasAgg
#imports for hamza code ends--#



class Attr:
    def __init__(self, property):
        self.property = property
        self.pos = 0
        self.neg = 0

attributes = [Attr('clothes'),Attr('quality'),Attr('price'),Attr('ambiance'),Attr('service'),Attr('staff'),Attr('store'),Attr('sale')]
pw = 0
nw = 0

#------------------------------------------------private methods-----------------------------------------------------------------    
#removing punctuation    
def RemovePunctuation(reviews):
    for index, row in reviews.iterrows():
        review = row['Review']
        review = re.sub("[\.!{2,}]", ' ', review)
        review = re.sub("[^a-zA-Z' ]+", '', review).lower()
        reviews.at[index, 'Review'] = review

#removal of words not in dictionary
def RemovingNonEnglishWords(reviews):
    for index, row in reviews.iterrows():
        review = row['Review']
        review = ' '.join([w for w in str(review).split() if wordnet.synsets(w)])
        reviews.at[index, 'Review'] = review

#get a list of stopWords    
def GetStopWords():
    oldStopWords = stopwords.words('english')
    exceptions = ['no', 'nor', 'not','don', "don't", 't', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stopWords = [word for word in oldStopWords if word not in exceptions]
    return stopWords
    
#removing stop words
def RemoveStopWords(reviews):
    stopWords = GetStopWords()
    for index, row in reviews.iterrows():
        review = row['Review']
        wordList = review.split() 
        filteredWords = [word for word in wordList if word not in stopWords]
        review = ' '.join(filteredWords)
        reviews.at[index, 'Review'] = review
#------------------------------------------------private methods end-----------------------------------------------------------------    

#print reviews
def PrintReviews(reviews):
    print(reviews)
    
#finding out the unique words
def UniqueWords(reviews):
    uniqueWords = list(reviews['Review'].str.split(' ', expand=True).stack().unique())
    print(len(uniqueWords))

#data cleansing
def CleanData(reviews):
    RemovePunctuation(reviews)
    RemovingNonEnglishWords(reviews)
    RemoveStopWords(reviews)
    
#cleaning an individual review
def CleanTestSentence(review):
    review = re.sub("[\.!{2,}]", ' ', review)
    review = re.sub("[^a-zA-Z' ]+", '', review).lower()
    stopWords = GetStopWords()
    wordList = review.split() 
    filteredWords = [word for word in wordList if word not in stopWords]
    review = ' '.join(filteredWords)
    wordsTokens = review.split()
    stemmedSentence = []
    porter = PorterStemmer()
    for word in wordsTokens:
        stemmedSentence.append(porter.stem(word))
        stemmedSentence.append(" ")
    review = "".join(stemmedSentence)
    return review

#Part of speech tagging
def part_of_speech_tagging(sentence):
    try:
        tokenized = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized)
        #print(tagged)
        phraseReg = r"""Phrase: {<RB.?>+<VB.?><NN.?> | <RB.?>*<VB.?>*<NNP>+<NN>? | <JJ.?>*<NN.?>+ | <JJ>*<VB>*<NN.?>}"""
        phraseParser = nltk.RegexpParser(phraseReg)
        #print(phraseParser)
        phrased = phraseParser.parse(tagged)
        #print(phrased)
        sentiment_analysis(phrased)
    except Exception as exp:
        print(str(exp))
        
def search_target(li):
    for i in li:
        if i[0] == 'clothes':
            return i[0]
        elif i[0] == 'quality':
            return i[0]
        elif i[0] == 'price':
            return i[0]
        elif i[0] == 'ambiance':
            return i[0]
        elif i[0] == 'service':
            return i[0]
        elif i[0] == 'staff':
            return i[0]
        elif i[0] == 'store':
            return i[0]
        elif i[0] == 'sale':
            return i[0]
        
from textblob import TextBlob
def phrasePolarity(phrase):
    global pw
    global nw
    polarity = TextBlob(phrase).sentiment.polarity
    if polarity >=0:
        pw = pw + 1
    else:
        nw = nw + 1
        
def sentiment_analysis(phrase):
    global pw
    global nw
    for i in phrase:
        pw = 0
        nw = 0
        target = search_target(i);
        for j in i: 
            if(j[1]== 'RB' or j[1]== 'RBR' or j[1]== 'RBS' or j[1]== 'JJ' or j[1]== 'JJR' or j[1]== 'JJS' or j[1]== 'VB' or j[1]== 'VBD' or j[1]== 'VBG' or j[1]== 'VBN' or j[1]== 'VBP' or j[1]== 'VBZ'):
                phrasePolarity(j[0])
        for k in attributes:
            if k.property == target:
                if pw > nw:
                    k.pos = k.pos + 1
                elif pw < nw:
                    k.neg = k.neg + 1
                    
def PlotGraph():
    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    # Example data
    labelsX = ('Clothes +', 'Clothes -', 'Quality +', 'Quality -', 'Price +', 'Price -', 'Ambiance +', 'Ambiance -', 'Service +', 'Service -', 'Staff +', 'Staff -', 'Store +', 'Store -', 'Sale +', 'Sale -' )
    y_pos = np.arange(len(labelsX))
    height = [attributes[0].pos, attributes[0].neg, attributes[1].pos, attributes[1].neg, attributes[2].pos, attributes[2].neg, attributes[3].pos, attributes[3].neg, attributes[4].pos, attributes[4].neg, attributes[5].pos, attributes[5].neg, attributes[6].pos, attributes[6].neg, attributes[7].pos, attributes[7].neg]
    # plt.barh(y_pos, height)


    plot = ax.barh(y_pos, height)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labelsX)
    ax.invert_yaxis()
    ax.set_xlabel('Number of reviews')
    ax.set_title('Feature Extraction')
    plot[0].set_color('g')
    plot[1].set_color('r') 
    plot[2].set_color('g')
    plot[3].set_color('r')
    plot[4].set_color('g')
    plot[5].set_color('r')
    plot[6].set_color('g')
    plot[7].set_color('r')
    plot[8].set_color('g')
    plot[9].set_color('r')
    plot[10].set_color('g')
    plot[11].set_color('r')
    plot[12].set_color('g')
    plot[13].set_color('r')
    plot[14].set_color('g')
    plot[15].set_color('r')
    response = HttpResponse(content_type = 'image/png')
    canvas = FigureCanvasAgg(plt.figure())
    canvas.print_png(response)
    return response








    # plt.show()
    
def InitializeAttributes():
    for t in attributes:
        t.pos = 0
        t.neg = 0
        
def RunProgram(reviews):
    InitializeAttributes()
    for index, row in reviews.iterrows():
        review = row['Review']
        if (review == ''):
            continue
        part_of_speech_tagging(review)
        fig=PlotGraph()
        return fig
	   
    



































# Create your views here.
def index(request):
	pd.set_option('display.max_colwidth', -1)
	# firebase = firebase.FirebaseApplication('https://markan-36b3b.firebaseio.com', None)
	# file=open(os.path.join(settings.BASE_DIR,'markan-36b3b-f5e3fe08a56d.json'))
	if (not len(firebase_admin._apps)):
		cred = credentials.Certificate('markan-36b3b-f5e3fe08a56d.json') #file stored in directory
		firebase_admin.initialize_app(cred)

	db = firestore.client()
	users_ref = db.collection(u'Comments')
	docs = users_ref.get()

	for doc in docs:
	    print(u'{} => {}'.format(doc.id, doc.to_dict()))

	docs = users_ref.get()
	df_text = pd.DataFrame() 
	count = 0
	# geo_point = firebase_admin.firestore.GeoPoint()
	for doc in docs:
	#     print('{}'.format(doc.to_dict()))
	    row = doc.to_dict()
	    df_text.at[count,'comment'] = row['comment']
	    df_text.at[count,'latitude'] = row['location'].latitude
	    df_text.at[count,'longitude'] = row['location'].longitude
	    df_text.at[count,'rating'] = row['rating']

	    count +=1

	# code for word-cloud
	word_cloud=''
	for row in df_text.itertuples():
   		word_cloud=word_cloud+row.comment

	# code ends
	df_map=df_text[['latitude', 'longitude']]
	df_map=df_map.values.tolist()

	df_text['rating'] = np.where(df_text['rating'] > 3, 'positive','negative')

	df_text.loc[(df_text['rating'] == 3),'rating'] = 'neutral'
	
	#Store details
	users_ref = db.collection(u'Stores')
	docs = users_ref.get()

	for doc in docs:
 		print(u'{} => {}'.format(doc.id, doc.to_dict()))
	docs = users_ref.get()
	df_store = pd.DataFrame() 
	count = 0

	for doc in docs:

	    row = doc.to_dict()
	    df_store.at[count,'id'] =doc.id
	    df_store.at[count,'name'] = row['name']
	    df_store.at[count,'logo'] = row['logo']
	    df_store.at[count,'image'] = row['image']
	    df_store.at[count,'details'] = row['details']
	    df_store.at[count,'tagline'] = row['tagline']
	    
	    
	    count +=1

	
	df_store=df_store.loc[df_store['id'] == 'CvVVqnL95aU1mzFZovfX'] 
	df_store_loc=df_store[['id','name','details','tagline']]

	#store details end

	reviews = pd.read_csv('C:/Users/Saad/workspace/workspace-for-social-media-analytics-app/RestaurantReviews.csv')
	CleanData(reviews)
	plot=RunProgram(reviews)
	# graph_div = plotly.offline.plot(fig, auto_open = False, output_type="div")

	












	# barcharts
	df_barchart=df_text.groupby('rating')['rating'].count()
	df_barchart=df_barchart.to_frame()

	# df_barchart=df_barchart.rename(index=str, columns={"rating": "count"})
	# df_barchart=df_barchart.reset_index(level=0)
	df_barchart=df_barchart.to_html(classes="barchart_dt table")
	store_image=df_store.iloc[0]['image']
	store_logo=df_store.iloc[0]['logo']

	range_gmaps=range(len(df_map))
	df_store_loc=df_store_loc.to_html(classes="table table-striped full-view-table table-responsive")
	df_text=df_text.to_html(classes="table table-striped full-view-table table-responsive")
	args={'df_text':df_text,'df_barchart': df_barchart,'word_cloud':word_cloud,'df_map':df_map,'range_gmaps':range_gmaps,
	'df_store':df_store_loc,'store_image':store_image,'store_logo':store_logo,'plot':plot}

	return render(request,'home.html',args)