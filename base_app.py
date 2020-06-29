"""
	This streamlit web app utilises various models to analyse 
	the sentiment of tweets regarding climate change and 
	classify them into one of the following classes: 
	2: The tweet is a message of factual news
	1: The author believes in climate change
	0: The message has a neutral sentiment
	-1: The author does not believe in climate change

	The web app also shows Exploratory Data Analysis and Insights of
	the given data.
"""


#---------------------------------------------------------------
# Streamlit dependencies

import streamlit as st

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import string
import pickle
import joblib
import re


#stopwords
stop = {'a',
 'about','above', 'after',
 'again', 'against',
 'ain', 'all',
 'am', 'an',
 'and', 'any',
 'are', 'aren',
 "aren't", 'as',
 'at', 'be',
 'because', 'been',
 'before', 'being',
 'below', 'between',
 'both', 'but',
 'by', 'can',
 'couldn', "couldn't",
 'd', 'did',
 'didn', "didn't",
 'do', 'does',
 'doesn', "doesn't",
 'doing', 'don',
 "don't", 'down',
 'during', 'each',
 'few', 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 'her',
 'here',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves'}

from collections import defaultdict

#loading the data
dftrain = pd.read_csv('train.csv')
# data preprocessing
raw = dftrain.head().copy()
# creating object for vectorizing the tweets
X= np.array(dftrain['message'])
y= np.array(dftrain['sentiment'])

# create vectorizer object
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer.fit(X_train)
# punctuation count
dftrain['punct_count']  = dftrain['message'].apply(lambda x: len([i for i in x if i in string.punctuation]))
# word count
word_count = dftrain['message'].apply(lambda x: len(x.split()))
dftrain['word_count'] = word_count
# punctuation removing function
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
# the following code is a url remover
def  clean_text(df, text_field):
    """
    this function takes in a dataframe,text field and removes urls from the text field then return a dataframe with urls removed form the text field
    """
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    return df

dftrain['message'] = clean_text(dftrain,'message')['message']

# tag words removing function
dftrain['message'] = dftrain['message'].apply(lambda x: (x.lower()).replace('climate change',''))
dftrain['message'] = dftrain['message'].apply(lambda x: (x.lower()).replace('global warming',''))


# changing background colour
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css('style.css')


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# dictionary of predictions and definitions
	pred_values = {2:'This is news',
	1:'The author believes in climate change',
	0:'The tweet is neutral in regards to climate change',
	-1:'The author does not believe in climate change'
	}
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Exploratory Data Analysis and Insights", "Prediction", 'Our People' ]
	selection = st.sidebar.radio("Choose Option", tuple(options))


	# Building out the "Information" page
	if selection == "Information":
		image = Image.open('markus-spiske-rxo6paehyqq-unsplash.jpg')
		st.info("General Information")
		st.image(image, caption='Climate change, photo by Markus Spiske' )#, use_column_width=True)
	
		# You can read a markdown file from supporting resources folder
		st.markdown("The purpose of this web app is to demonstrate the functionality and performance \n of various models on tweet analysis and classification specifically for climate change. And to draw insights that could be beneficial for the client.")

		st.subheader("View of raw Data structure")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw) # will write the df to the page
		

	# Building out the predication page
	if selection == "Prediction":
		# option to choose model
		modsel = st.selectbox('Choose a model for prediction:', ["SKLearn Pipeline", "Logistic Regression", "SVM"])
		# Load the model from the file 
		 
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		if st.button("Classify"):
			if modsel == "SKLearn Pipeline":
				skppl = joblib.load('skppl2.pkl')
				prediction =skppl.predict([tweet_text])
				st.success("Text Category: {}".format(pred_values[prediction[0]]))
			elif modsel == "Logistic Regression":
				lr_vect = joblib.load('lr_vect.pk')
				lr_model = joblib.load('lr_mod.pkl')
				tweet_x = np.array([tweet_text])
				x_ = lr_vect.transform(tweet_x)

				prediction =lr_model.predict(x_)
				st.success("Text Category: {}".format(pred_values[prediction[0]]))

			elif modsel == "SVM":
				svm_vecty = joblib.load('svm_vecty.pk')
				svm_model = joblib.load('svm_model.pkl')
				tweet_text = remove_punctuations(tweet_text)
				tweet_x = np.array([tweet_text])
				x_ = svm_vecty.transform(tweet_x)
				prediction =svm_model.predict(x_)
				st.success("Text Category: {}".format(pred_values[prediction[0]]))
		#	# You can use a dictionary or similar structure to make this output
		#	# more human interpretable.
		
	if selection == "Exploratory Data Analysis and Insights":
		st.title("Exploratory Data Analysis")
		# boxplots for word count analysis
		# create subplots
		st.subheader("Boxplots")
		plt.figure(figsize=(1,1))

		fig, axs = plt.subplots(1, 4, sharey = True)
		fig.suptitle('Boxplots for word count of each class')

		# class 2 plot
		y2 = dftrain[dftrain['sentiment'] == 2]['word_count']
		axs[0].boxplot(y2)
		axs[0].set_xlabel('class 2')

		# class 1 plot
		y1 = dftrain[dftrain['sentiment'] == 1]['word_count']
		axs[1].boxplot(y1)
		axs[1].set_xlabel('class 1')

		# class 0 plot
		y0 = dftrain[dftrain['sentiment'] == 0]['word_count']
		axs[2].boxplot(y0)
		axs[2].set_xlabel('class 0')

		# class -1 plot
		y_1 = dftrain[dftrain['sentiment'] == -1]['word_count']
		axs[3].boxplot(y_1)
		axs[3].set_xlabel('class -1')

		axs[0].set_ylabel('Word Count')

		st.pyplot()
		
		st.markdown('The boxplots of word count show distinct properties for each class.\n The presence of outliers, varying medians and range sizes imply \n that the word count property will add substantial value to model training.')
		st.subheader("Bar chart")
		#A bar graph comparing the frequency of each sentiment
		dftrain['sentiment'].value_counts().plot(kind = 'bar')
		plt.xticks(rotation='horizontal')
		plt.xlabel('Sentiments')
		plt.ylabel('Sentiment counts')
		plt.title('Sentiment Value Counts')
		st.pyplot()
		st.markdown('This graph shows that these four classes are imbalanced, which affects the accuracy of the model negatively. This shows that resambling is necessary before training a model with this data.')

		# the following code focuses on word clouds
		# extracting messages of each class
		st.subheader("Wordclouds")
		class2_words = ' '.join([text for text in dftrain[dftrain['sentiment']==2]['message']])
		class1_words = ' '.join([text for text in dftrain[dftrain['sentiment']==1]['message']])
		class0_words = ' '.join([text for text in dftrain[dftrain['sentiment']==0]['message']])
		class_neg1_words = ' '.join([text for text in dftrain[dftrain['sentiment']==-1]['message']])

		# creating wordclouds for each class
		wordcloud2 = WordCloud(width=800, height=500,random_state=21,max_font_size=110).generate(class2_words)
		wordcloud1 = WordCloud(width=800, height=500,random_state=21,max_font_size=110).generate(class1_words)
		wordcloud0 = WordCloud(width=800, height=500,random_state=21,max_font_size=110).generate(class0_words)
		wordcloudneg1 = WordCloud(width=800, height=500,random_state=21,max_font_size=110).generate(class_neg1_words)
		fig = plt.figure(figsize=(1000,500))
		fig,axs = plt.subplots(2, 2)
		fig.suptitle('Boxplots for word count of each class')
		# word cloud plots
		axs[0,0].imshow(wordcloud2, interpolation="bilinear")
		axs[1,1].imshow(wordcloud1, interpolation="bilinear")
		axs[0,1].imshow(wordcloud0, interpolation="bilinear")
		axs[1,0].imshow(wordcloudneg1, interpolation="bilinear")
		# removing axes
		axs[0,0].axis('off')
		axs[1,1].axis('off')
		axs[0,1].axis('off')
		axs[1,0].axis('off')
		# word cloud titles
		axs[0,0].set_title('Climate Change News')
		axs[1,1].set_title('Pro Climate Change')
		axs[0,1].set_title('Neutral to Climate Change')
		axs[1,0].set_title('Anti Climate Change')
		st.pyplot()
		st.markdown('These wordclouds show each class after removal of the two phrases: "global warming" and "climate change".The key take away from these word clouds is that each class has its own distinct predominant words(or phrases), and thus the removal of those tag words makes the classes more distinct.')
		
		# -------------------------------------------------
		# the corpus
		st.subheader("Word count distribution plots")
		image2 = Image.open('WhatsApp Image 2020-06-29 at 15.50.26.jpeg')
		st.image(image2)
		st.markdown('This graph shows the distribution of the word count frequency. It shows the relationship between the number of words in a tweet and it''s class. From the graph we can see that the tweets from class News have a lenth from 5 to 25 and it''s tweet lenth has normal distribution along mean 16. It also shows the same patten for the rest of the classes. With class Pro having work count lenths from 4 till 30, with more tweets around lenth 21. Class Neutral has lenths from 2 to 30, with most tweets around 20. And Class Anti has lenths from 4 to 30, with most tweets around 21.')

		st.title("Insights")
		st.markdown('In conclusion we chose the Pipeline Model because the score was the highest. This model will be able to classify whether or not a person believes in climate change, based on their novel tweet data. Three Models score: ')
		st.markdown('Pipeline Model:  0.75260')
		st.markdown('Logistic Regression Model: 0.74952')
		st.markdown('SVM Model: 0.64615')


	if selection == 'Our People':
		st.markdown('Project Owner: EDSA')
		st.markdown('Our Data Scientists Team')
		st.markdown('Scrum Master : Noluthando Khumalo')
		st.markdown('Developer: Itumeleng Ngoetjana')
		st.markdown('Designer : Thavha Tsiwana')
		st.markdown('Designer : Pontsho Mokone')
		st.markdown('Tester : Tumelo Mokubi')


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
