"""
	This streamlit web app utilises various models to analyse 
	the sentiment of tweets regarding climate change and 
	classify them into one of the following classes: 
	2: The tweet is a message of factual news
	1: The author believes in climate change
	0: The message has a neutral sentiment
	-1: The author does not believe in climate change
"""
#---------------------------------------------------------------
# Streamlit dependencies
import streamlit as st

# Data dependencies
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import string
# Enter your code here:
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')

# splitting the model
from sklearn.model_selection import train_test_split

X = dftrain['message']  # this time we want to look at the text
y = dftrain['sentiment']
#####################Pipeline Model
# word count analysis
word_count = dftrain['message'].apply(lambda x: len(x.split()))
dftrain['word_count'] = word_count

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# since i am going to use the tfidtransformer and linearsvc, i should thus create their objects first

# create tfidf object
from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_transformer = TfidfTransformer()

# create linearsvc object
from sklearn.svm import LinearSVC
# clf = LinearSVC()

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  

############################Logistic Regression Model
#creating a copy of our dataframe
# punctuation count 

dftrain['punct_count']  = dftrain['message'].apply(lambda x: len([i for i in x if i in string.punctuation]))

df_copy = dftrain

# setting our XFeature and label variables
X = df_copy[['word_count','punct_count']]  
y = df_copy['sentiment']

# splitting the train data
from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#selecting the logistic regression model from sklearn and selecting lbfgs as our solver
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='lbfgs')

#training our model
lr_model.fit(x_train, Y_train)

##########################SVM Model
df_SVM_train = dftrain
df_SVM_test=dftest

#Removing puncuations from the messege column
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df_SVM_train["message"] = df_SVM_train['message'].apply(remove_punctuations)
df_SVM_test["message"] = df_SVM_test['message'].apply(remove_punctuations)

#Defining training and testing data
x_train = df_SVM_train['message']
y_train = df_SVM_train['sentiment']
x_test = df_SVM_test['message']

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(x_train)

#Transforming also the x_test data to numerical data
X_test_counts = count_vect.transform(x_test)

from sklearn.svm import SVC  # to be added on the import cell
svm = SVC()
svc = SVC(kernel='linear',C=2,degree=5,gamma='auto')
svc.fit(X_train_counts, y_train)

################################

# changing background colour
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css('style.css')

@st.cache(suppress_st_warning=True)

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
	options = ["Information", "Exploratory Data Analysis", "Prediction", 'Test Shop', 'Our People' ]
	selection = st.sidebar.radio("Choose Option", tuple(options))


	# Building out the "Information" page
	if selection == "Information":
		image = Image.open('images\markus-spiske-rxo6paehyqq-unsplash.jpg')
		st.info("General Information")
		st.image(image, caption='Climate change, photo by Markus Spiske' )#, use_column_width=True)
	
		# You can read a markdown file from supporting resources folder
		st.markdown("The purpose of this web app is to demonstrate the functionality and performance \n of various models on tweet analysis and classification specifically for climate change.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(dftrain[['sentiment','message','tweetid']].head(10)) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		# option to choose model
		modsel = st.selectbox('Choose a model for prediction:', ["SKLearn Pipeline", "Logistic Regression", "SVM"])
		

		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			if modsel == "SKLearn Pipeline":
				prediction =text_clf.predict([tweet_text])
				st.success("Text Category: {}".format(pred_values[prediction[0]]))
			elif modsel == "Logistic Regression":
				prediction =lr_model.predict(tweet_text)
				st.success("Text Category: {}".format(pred_values[prediction[0]]))
			elif modsel == "SVM":
				prediction =svc.predict(tweet_text)
				st.success("Text Category: {}".format(pred_values[prediction[0]]))
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
	if selection == "Exploratory Data Analysis":
		# boxplots for word count analysis
		# create subplots
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
		st.markdown('The key take away from these word clouds is that each class has its own distinct predominant words(or phrases)')

	if selection == 'Test Shop':
		st.markdown('In this section im just testing things out,\n dont know what i should put in and how i should do it but it will be here for now')

		# testing selectbox
		st.markdown('Testing selectbox')
		selbox = ['a','b','x']
		sb = st.selectbox('choose', selbox)
		st.write(sb)

		# testing the csv uploader
		uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
		if uploaded_file is not None:
			dat1 = pd.read_csv(uploaded_file)
			st.write(dat1.head())

		# testing images
		image = Image.open('images\markus-spiske-rxo6PaEhyqQ-unsplash.jpg')

		st.image(image, caption='Climate change, photo by Markus Spiske' , use_column_width=True)
	if selection == 'Our People':
		st.markdown('Project Owner: EDSA')
		st.markdown('Scrum Master : Noluthando Khumalo')
		st.markdown('Developer: Itumeleng Ngoetjana')
		st.markdown('Designer : Thavha Tsiwana')
		st.markdown('Designer : Pontsho Mokone')
		st.markdown('Tester : Tumelo Mokubi')


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
