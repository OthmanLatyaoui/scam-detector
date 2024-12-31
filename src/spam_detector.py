import pandas as pd
import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download("all")

tokenizer = RegexpTokenizer(r'\w+')
test = "Ceci est un exemple de texte à tokeniser. Il contient des mots en anglais et en français!"
tokens = tokenizer.tokenize(test)

stop_french_english = stopwords.words('french')
stop_french_english.extend(stopwords.words('english'))

print("Voici les tokens:", tokens)
print("Stopwords:", stop_french_english)

exclude_ponctuation = set(string.punctuation)
stop_french_english.extend(exclude_ponctuation)

text2 = "This is an example of text to tokenize. It contains words in English and French !"

lemma = WordNetLemmatizer()

tokens2 = tokenizer.tokenize(text2.lower())
print("Voici les tokens :", tokens2)

tokens_filtre = [token for token in tokens2 if token.lower() not in stop_french_english]
print(tokens_filtre)

lemma.lemmatize("running", pos='v')

def preprocess_text(text):

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.lower() not in stop_french_english]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(token, pos='a'), pos='v'), pos='n') for token in filtered_tokens]
    result_text = ' '.join(lemmatized_tokens)

    return result_text

text = "Arrived totally in as between private. Favour of so as on pretty though elinor direct. Reasonable estimating be alteration we themselves entreaties me of reasonably. Direct wished so be expect polite valley. Whose asked stand it sense no spoil to. Prudent you too his conduct feeling limited and. Side he lose paid as hope so face upon be. Goodness did suitable learning put."
preprocess_text(text)

csv_filename = '/content/sample_data/spam.csv'

spam_df = pd.read_csv(csv_filename, encoding='latin1')
spam_df = spam_df.iloc[:, :-3]
num_lines = spam_df.shape[0]

print(spam_df.head())
print(f"Number of lines: {num_lines}")

first_column_counts = spam_df.iloc[:, 0].value_counts(normalize=True) * 100
print(first_column_counts)

spam_df.columns = ['label', 'message']

spam_messages = spam_df[spam_df['label'] == 'spam']['message']
ham_messages = spam_df[spam_df['label'] == 'ham']['message']


spam_string = ' '.join(spam_messages)
ham_string = ' '.join(ham_messages)
print(spam_string)


wordcloud = WordCloud(width = 800, height = 400, background_color ='white').generate(spam_string)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

spam_df['label'] = spam_df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(spam_df['message'], spam_df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"precision: {accuracy}")

def predict_spam(message):

    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return 'spam' if prediction[0] == 1 else 'ham'

new_message = "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"
print(f"Message: {new_message}")
print(f"Prediction: {predict_spam(new_message)}")