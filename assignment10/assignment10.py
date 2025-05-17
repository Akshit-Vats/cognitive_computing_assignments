import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
nltk.download('all')

text_q1 = "One topic I find incredibly compelling is space exploration. The idea that humans have sent spacecraft beyond our solar system, like Voyager 1, and are planning missions to Mars shows how far curiosity and engineering can take us. Each mission, whether it's landing a rover on the Moon or capturing images of distant galaxies with the James Webb Telescope, pushes the boundaries of what we know. It’s humbling to realize how small Earth is in the vastness of space, yet we continue to reach out with questions and tools."

text_lower = text_q1.lower()
text_cleaned = re.sub(r'[^\w\s]', '', text_lower)

sentences = sent_tokenize(text_cleaned)
words_nltk = word_tokenize(text_cleaned)
words_split = text_cleaned.split()

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words_nltk if word not in stop_words]

fdist = nltk.FreqDist(filtered_words)

words_only = re.findall(r'\b[a-zA-Z]+\b', text_cleaned)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

texts_q3 = [
    "AI-Powered Drones Now Assist in Disaster Relief Across Southeast Asia",
    "This smartwatch does everything I need—tracks my workouts, syncs with my phone, and the battery lasts nearly a week. Only downside? The strap gets uncomfortable after long use.",
    "The latest sci-fi series on streaming feels like a breath of fresh air—tight storytelling, stunning visuals, and characters that actually evolve. It’s rare to see such balance between action and emotion."
]
cv = CountVectorizer()
bow_matrix = cv.fit_transform(texts_q3)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(texts_q3)
keywords = [tfidf.get_feature_names_out()[idx] for idx in tfidf_matrix.toarray().argsort(axis=1)[:, -3:]]

text1 = "Artificial Intelligence enables machines to mimic human intelligence, learning from data to make decisions or predictions. It powers technologies like voice assistants, image recognition, and recommendation systems. AI adapts and improves over time, making it incredibly useful in fields like healthcare, finance, and transportation. Its biggest strength lies in automating complex tasks with speed and accuracy."
text2 = "Blockchain is a decentralized digital ledger that records transactions across multiple computers. It ensures transparency, security, and immutability without the need for a central authority. Commonly used in cryptocurrencies like Bitcoin, it’s also expanding into areas like supply chain, voting systems, and digital identity. Its main appeal lies in trust and tamper-proof data handling."
words1 = set(word_tokenize(re.sub(r'[^\w\s]', '', text1.lower())))
words2 = set(word_tokenize(re.sub(r'[^\w\s]', '', text2.lower())))
jaccard_similarity = len(words1 & words2) / len(words1 | words2)

texts_tfidf = TfidfVectorizer()
vectors = texts_tfidf.fit_transform([text1, text2])
cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])

review = "I’ve been using the Anker Soundcore speaker for a few weeks now, and it’s honestly impressive for its size and price. The sound is rich and clear, with surprisingly deep bass for such a compact device. Bluetooth connection is stable, and the battery easily lasts over 20 hours on a single charge. It’s also lightweight and waterproof, making it perfect for outdoor use. Overall, a great value for casual music lovers or travel use."
blob = TextBlob(review)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity
wc = WordCloud().generate(review)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

text_train = "Artificial Intelligence (AI) is rapidly changing the way we live and work, from personalized recommendations on streaming platforms to advanced medical diagnostics. At its core, AI uses data and algorithms to mimic human intelligence, learning patterns and making decisions. What sets modern AI apart is its ability to improve over time through experience, making it ideal for tasks like language translation, fraud detection, and even creative writing. However, as AI becomes more integrated into society, it also raises concerns around bias, privacy, and ethical use. Balancing innovation with responsibility is key to unlocking its full potential."
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_train])
sequences = []
words = text_train.split()

for i in range(1, len(words)):
    seq = words[:i + 1]
    tokenized_seq = tokenizer.texts_to_sequences([' '.join(seq)])[0]
    sequences.append(tokenized_seq)

padded = pad_sequences(sequences)

model = Sequential()
model.add(Embedding(input_dim=50, output_dim=10, input_length=padded.shape[1]))
model.add(LSTM(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()
