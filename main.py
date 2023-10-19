import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import KFold
import shap
import json
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from langdetect import detect

class SentimentAnalysis:

    def __init__(self, model_path=None, model_type='basic', tokenizer_path=None, embedding_path=None, optimizer_path=None):
        if model_path:
            self.model = load_model(model_path)
            if tokenizer_path:
                with open(tokenizer_path, 'r') as f:
                    self.tokenizer = Tokenizer.from_json(f.read())
            else:
                self.tokenizer = None
            if optimizer_path:
                self.optimizer = tf.keras.optimizers.load(optimizer_path)
            else:
                self.optimizer = 'adam'
        else:
            self.model = None
            self.model_type = model_type
            self.tokenizer = None
            self.embedding_layer = None
            self.embedding_type = None
            self.hyperparameters = None
            if embedding_path:
                self.load_embedding(embedding_path)

    def load_embedding(self, embedding_path):
        # Load pre-trained embeddings (e.g., Word2Vec or BERT)
        if "word2vec" in embedding_path:
            # Load Word2Vec embeddings
            word2vec = Word2Vec.load(embedding_path)
            vocab_size = len(word2vec.wv.vocab)
            embedding_matrix = np.zeros((vocab_size, word2vec.vector_size))
            for word, i in self.tokenizer.word_index.items():
                if word in word2vec.wv:
                    embedding_matrix[i - 1] = word2vec.wv[word]
            self.embedding_layer = keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=word2vec.vector_size,
                weights=[embedding_matrix],
                input_length=self.model.input_shape[1],
                trainable=False
            )
            self.embedding_type = 'word2vec'
        elif "bert" in embedding_path:
            # Load pre-trained BERT model
            self.tokenizer = BertTokenizer.from_pretrained(embedding_path)
            self.embedding_type = 'bert'
        else:
            raise ValueError("Unsupported embedding type")

    def create_model(self, vocab_size, embedding_dim=16, max_length=100, num_classes=2):
        model = keras.Sequential()
        if self.embedding_layer:
            model.add(self.embedding_layer)
        else:
            model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_data(self, X_train, y_train, vocab_size=None, max_length=100):
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')

        if self.model_type == 'basic':
            num_classes = 2
        elif self.model_type == 'multi-class':
            num_classes = len(set(y_train))
        y_train = keras.utils.to_categorical(y_train, num_classes)

        return X_train, y_train

    def train(self, X_train, y_train, vocab_size=None, embedding_dim=16, max_length=100, epochs=5, hyperparameters=None, early_stopping=None):
        if hyperparameters:
            self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = self.tune_hyperparameters(X_train, y_train, max_length)

        X_train, y_train = self.preprocess_data(X_train, y_train, vocab_size, max_length)

        if not self.model:
            vocab_size = max(self.tokenizer.word_index.values()) + 1
            self.model = self.create_model(vocab_size, embedding_dim, max_length, len(y_train[0]))

        callbacks = []
        if early_stopping:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=early_stopping['monitor'], patience=early_stopping['patience'])
            callbacks.append(early_stopping_callback)

        self.model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks)

    def fine_tune_bert(self, X_train, y_train, max_length=100, epochs=5, learning_rate=2e-5, batch_size=32):
        X_train, y_train = self.preprocess_data(X_train, y_train, max_length=max_length)
        num_classes = len(y_train[0])

        if not self embedding_type == 'bert':
            raise ValueError("This method is only applicable to BERT embeddings.")

        # Load pre-trained BERT model
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
        tokenizer = self.tokenizer

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.CategoricalAccuracy('accuracy')

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # Tokenize the input data
        inputs = tokenizer(X_train, padding=True, truncation=True, return_tensors='tf', max_length=max_length)
        dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), y_train)).batch(batch_size)

        # Fine-tune BERT model
        history = model.fit(dataset, epochs=epochs)

        return history

    def predict(self, text):
        text = [text]
        text = self.tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, maxlen=self.model.input_shape[1], padding='post', truncating='post')

        prediction = self.model.predict(text)
        return prediction[0]

    def batch_predict(self, texts):
        texts = self.tokenizer.texts_to_sequences(texts)
        texts = pad_sequences(texts, maxlen=self.model.input_shape[1], padding='post', truncating='post')
        predictions = self.model.predict(texts)
        return predictions

    def evaluate(self, X_test, y_test):
        X_test, y_test = self.preprocess_data(X_test, y_test)
        y_pred = self.model.predict(X_test)
        y_pred_classes = [1 if pred[0] > 0.5 else 0 for pred in y_pred]

        accuracy = accuracy_score(y_test[:, 1], y_pred_classes)
        report = classification_report(y_test[:, 1], y_pred_classes, target_names=['Negative', 'Positive'])

        return accuracy, report

    def save_model(self, model_path, tokenizer_path=None, optimizer_path=None):
        self.model.save(model_path)
        if tokenizer_path:
            with open(tokenizer_path, 'w') as f:
                f.write(self.tokenizer.to_json())
        if optimizer_path:
            self.optimizer.save(optimizer_path)

    def load_model(self, model_path, tokenizer_path=None, optimizer_path=None):
        self.model = load_model(model_path)
        if tokenizer_path:
            with open(tokenizer_path, 'r') as f:
                self.tokenizer = Tokenizer.from_json(f.read())
        if optimizer_path:
            self.optimizer = tf.keras.optimizers.load(optimizer_path)

    def extract_important_words(self, text, n_words=10):
        # Get the top n_words that contribute to the positive prediction
        text_seq = self.tokenizer.texts_to_sequences([text])
        word_ids = text_seq[0]
        word_scores = self.model.layers[-1].get_weights()[0]
        important_word_ids = [word_ids[i] for i in np.argsort(-word_scores[word_ids])[:n_words]]
        important_words = [word for word, word_id in self.tokenizer.word_index.items() if word_id in important_word_ids]

        return important_words

    def hyperparameter_tuning(self, X, y):
        # Hyperparameter tuning using Bayesian optimization
        param_space = {
            'n_estimators': (10, 200),
            'max_depth': (1, 32),
            'min_samples_split': (0.1, 1.0),
            'min_samples_leaf': (0.1, 0.5),
        }
        opt = BayesSearchCV(
            RandomForestClassifier(),
            param_space,
            n_iter=32,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            return_train_score=False
        )
        opt.fit(X, y)
        best_params = opt.best_params_
        best_score = opt.best_score_

        return best_params, best_score

    def explain_prediction(self, text):
        # Explain a model's prediction using LIME
        explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
        explanation = explainer.explain_instance(text, self.predict, num_features=10)

        return explanation

    def cross_validate(self, X, y, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

            self.train(X_train, y_train, epochs=5)  # Example: 5 epochs, you can adjust this
            accuracy, _ = self.evaluate(X_test, y_test)
            accuracies.append(accuracy)

        return accuracies

    def shapley_explain(self, X, y, num_samples=100):
        # Explain model predictions using Shapley values
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer.shap_values(X[:num_samples])

        return shap_values

    def ensemble_models(self, models, weights=None):
        # Create an ensemble of models
        ensemble = VotingClassifier(estimators=[(f'model_{i}', model) for i, model in enumerate(models)], voting='soft', weights=weights)
        ensemble.fit(X_train, y_train)
        accuracy = ensemble.score(X_test, y_test)
        return ensemble, accuracy

    def preprocess_text(self, text, use_gpt3=False):
        if use_gpt3:
            # Use GPT-3 for sentiment analysis
            generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
            result = generator(text, max_length=50, num_return_sequences=1)
            return result[0]['generated_text']
        else:
            # Perform regular preprocessing
            text = [text]
            text = self.tokenizer.texts_to_sequences(text)
            text = pad_sequences(text, maxlen=self.model.input_shape[1], padding='post', truncating='post')
            return text

    def model_averaging(self, models, X_test):
        # Model averaging for a list of models
        predictions = [model.predict(X_test) for model in models]
        avg_predictions = np.mean(predictions, axis=0)
        return avg_predictions

    def batch_evaluate(self, X_test, y_test):
        X_test, y_test = self.preprocess_data(X_test, y_test)
        y_pred = self.model.predict(X_test)
        y_pred_classes = [1 if pred[0] > 0.5 else 0 for pred in y_pred]

        accuracy = accuracy_score(y_test[:, 1], y_pred_classes)
        f1 = f1_score(y_test[:, 1], y_pred_classes)
        roc_auc = roc_auc_score(y_test[:, 1], y_pred_classes)

        return accuracy, f1, roc_auc

    def interactive_word_cloud(self, text):
        # Generate an interactive word cloud for text visualization
        wordcloud = WordCloud(width=800, height=400, max_words=100).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def text_preprocessing(self, text, clean=False, normalize=False):
        # Text data preprocessing
        if clean:
            # Clean the text (e.g., remove special characters)
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters

        if normalize:
            # Normalize the text (e.g., remove stop words)
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            text = ' '.join([word for word in word_tokens if word not in stop_words])

        return text

    def detect_language(self, text):
        # Detect the language of the input text
        try:
            lang = detect(text)
            return lang
        except:
            return "unknown"

    def detect_sentiment(self, text, language):
        # Detect sentiment in text based on the detected language
        if language == "en":
            sentiment = self.predict(text)
            return sentiment
        else:
            return "Sentiment analysis for non-English languages is not supported."

    def analyze_multimodal_data(self, text_data, image_data):
        # Analyze sentiment in multimodal data (text and image)
        text_sentiment = self.predict(text_data)
        # You can add image sentiment analysis here if you have an image model
        return text_sentiment

    def integrate_with_chatbot(self, text):
        # Integrate with a chatbot platform to analyze sentiment and provide chatbot responses
        # Example: Call a chatbot API with text and return the sentiment along with the chatbot response
        response = requests.post('chatbot-api-url', data={'text': text})
        chatbot_response = response.text
        sentiment = self.predict(text)
        return sentiment, chatbot_response

    def detect_anomalies(self, text_data):
        # Detect anomalies in text data using Isolation Forest
        text_data = self.tokenizer.texts_to_sequences(text_data)
        text_data = pad_sequences(text_data, maxlen=self.model.input_shape[1], padding='post', truncating='post')
        anomaly_detector = IsolationForest()
        anomalies = anomaly_detector.fit_predict(text_data)
        return anomalies

if __name__ == "__main__":
    # Example usage
    sentiment_analysis = SentimentAnalysis(model_type='basic')

    # Example training data
    X_train = ["This is a positive sentence.", "This is a negative sentence.", "This is a neutral sentence"]
    y_train = [1, 0, 1]  # 0 for negative, 1 for positive

    # Training the model with early stopping
    early_stopping_params = {'monitor': 'val_accuracy', 'patience': 2}
    history = sentiment_analysis.train(X_train, y_train, vocab_size=10000, embedding_dim=16, max_length=100, epochs=5, early_stopping=early_stopping_params)

    # Fine-tune BERT model
    bert_history = sentiment_analysis.fine_tune_bert(X_train, y_train, max_length=100, epochs=5)

    # Save the model, tokenizer, and optimizer
    sentiment_analysis.save_model("sentiment_model.h5", "tokenizer.json", "optimizer.h5")

    # Load the model, tokenizer, and optimizer
    sentiment_analysis.load_model("sentiment_model.h5", "tokenizer.json", "optimizer.h5")

    # Example prediction
    text_to_analyze = "This is a neutral sentence."
    prediction = sentiment_analysis.predict(text_to_analyze)
    print(f"Prediction: {prediction}")

    # Batch prediction
    texts_to_analyze = ["This is a positive sentence.", "This is a negative sentence."]
    batch_predictions = sentiment_analysis.batch_predict(texts_to_analyze)
    print("Batch Predictions:")
    for text, prediction in zip(texts_to_analyze, batch_predictions):
        print(f"Text: {text}, Prediction: {prediction}")

    # Evaluate the model
    X_test = ["This is a positive review.", "This is a negative review.", "This is a neutral review"]
    y_test = [1, 0, 1]
    accuracy, report = sentiment_analysis.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Extract and display important words for a prediction
    text_to_analyze = "This is a very positive review with great quality."
    important_words = sentiment_analysis.extract_important_words(text_to_analyze, n_words=5)
    print(f"Important Words: {important_words}")

    # Hyperparameter tuning for a RandomForest model
    X = ["This is a positive review.", "This is a negative review.", "This is a neutral review"]
    y = [1, 0, 1]
    best_params, best_score = sentiment_analysis.hyperparameter_tuning(X, y)
    print("Best Hyperparameters:", best_params)
    print("Best Accuracy Score:", best_score)

    # Explain a model's prediction
    text_to_explain = "This is a positive review."
    explanation = sentiment_analysis.explain_prediction(text_to_explain)
    explanation.show_in_notebook()

    # Cross-Validation
    accuracies = sentiment_analysis.cross_validate(X_train, y_train, k=5)
    print(f"Cross-Validation Accuracies: {accuracies}")

    # Explain model predictions using Shapley values
    shap_values = sentiment_analysis.shapley_explain(X_test, y_test)
    print(f"Shapley Values: {shap_values}")

    # Ensemble of models
    model1 = SentimentAnalysis()
    model2 = SentimentAnalysis()
    models = [model1, model2]
    weights = [0.5, 0.5]
    ensemble, ensemble_accuracy = sentiment_analysis.ensemble_models(models, weights)
    print(f"Ensemble Accuracy: {ensemble_accuracy}")

    # Preprocess text using GPT-3
    text_to_preprocess = "This is a review about a product."
    preprocessed_text = sentiment_analysis.preprocess_text(text_to_preprocess, use_gpt3=True)
    print(f"Preprocessed Text: {preprocessed_text}")

    # Model Averaging
    model3 = SentimentAnalysis()
    model4 = SentimentAnalysis()
    models_for_averaging = [model3, model4]
    X_test_for_averaging = ["This is a positive sentence.", "This is a negative sentence."]
    avg_predictions = sentiment_analysis.model_averaging(models_for_averaging, X_test_for_averaging)
    print("Model Averaging Predictions:")
    print(avg_predictions)

    # Batch Evaluation
    X_test_batch = ["This is a positive review.", "This is a negative review.", "This is a neutral review"]
    y_test_batch = [1, 0, 1]
    batch_accuracy, batch_f1, batch_roc_auc = sentiment_analysis.batch_evaluate(X_test_batch, y_test_batch)
    print(f"Batch Accuracy: {batch_accuracy}")
    print(f"Batch F1 Score: {batch_f1}")
    print(f"Batch ROC AUC Score: {batch_roc_auc}")

    # Interactive Word Cloud
    text_for_word_cloud = "This is a positive and wonderful review about a fantastic product."
    sentiment_analysis.interactive_word_cloud(text_for_word_cloud)

    # Text Data Preprocessing
    text_to_preprocess = "This is some text with punctuation, numbers, and stop words."
    cleaned_text = sentiment_analysis.text_preprocessing(text_to_preprocess, clean=True, normalize=False)
    print(f"Cleaned Text: {cleaned_text}")

    # Detect Language and Sentiment
    text_to_detect = "This is a review in English."
    language = sentiment_analysis.detect_language(text_to_detect)
    sentiment = sentiment_analysis.detect_sentiment(text_to_detect, language)
    print(f"Detected Language: {language}")
    print(f"Detected Sentiment: {sentiment}")

    # Analyze Multimodal Data
    text_data = "This is a positive review."
    image_data = "image_bytes_or_path"  # Replace with actual image data
    multimodal_sentiment = sentiment_analysis.analyze_multimodal_data(text_data, image_data)
    print(f"Multimodal Sentiment: {multimodal_sentiment}")

    # Integrate with Chatbot
    chatbot_text = "How can I help you?"
    sentiment, response = sentiment_analysis.integrate_with_chatbot(chatbot_text)
    print(f"Sentiment: {sentiment}")
    print(f"Chatbot Response: {response}")

    # Detect Anomalies
    text_data_to_detect = ["This is a normal review.", "This is an anomaly with unusual content."]
    anomalies = sentiment_analysis.detect_anomalies(text_data_to_detect)
    print(f"Anomalies: {anomalies}")
