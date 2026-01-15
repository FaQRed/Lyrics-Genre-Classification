import pandas as pd
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


Path("logs").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class MusicGenreClassifier:
    def __init__(self):
        self.config = {
            'max_features': 5000,
            'hidden_layers': (256, 128),
            'epochs': 30,  # Для фиксации кривых обучения
            'random_state': 42
        }
        self.le = LabelEncoder()
        self.tfidf = TfidfVectorizer(max_features=self.config['max_features'], stop_words='english')


        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.config['hidden_layers'],
            max_iter=1,
            warm_start=True, # add train model in cycle
            random_state=self.config['random_state']
        )


        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

    def load_data(self, train_path, test_path):
        logger.info("Loading data...")
        train = pd.read_csv(train_path).dropna(subset=['clean_text', 'Genre'])
        test = pd.read_csv(test_path).dropna(subset=['clean_text', 'Genre'])
        return train, test

    def train(self, train_df, val_df):
        logger.info("Preparing features...")
        X_train = self.tfidf.fit_transform(train_df['clean_text'])
        y_train = self.le.fit_transform(train_df['Genre'])

        X_val = self.tfidf.transform(val_df['clean_text'])
        y_val = self.le.transform(val_df['Genre'])

        classes = np.unique(y_train)

        logger.info(f"Starting training for {self.config['epochs']} epochs...")

        for epoch in tqdm(range(self.config['epochs'])):

            self.mlp.partial_fit(X_train, y_train, classes=classes)

            train_loss = self.mlp.loss_
            train_acc = self.mlp.score(X_train, y_train)

            val_acc = self.mlp.score(X_val, y_val)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

        self._plot_learning_curves()

    def _plot_learning_curves(self):
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Train Loss', color='red')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('reports/mlp_learning_curves.png')
        logger.info("Learning curves saved to reports/mlp_learning_curves.png")

    def evaluate(self, test_df):
        X_test = self.tfidf.transform(test_df['clean_text'])
        y_test = self.le.transform(test_df['Genre'])
        y_pred = self.mlp.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.le.classes_)

        logger.info(f"Final Test Accuracy: {acc:.4f}")
        print(f"\nReport:\n{report}")

        self._plot_confusion_matrix(y_test, y_pred)

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.le.classes_, yticklabels=self.le.classes_, cmap='Blues')
        plt.title('MLP Confusion Matrix')
        plt.savefig('reports/mlp_confusion_matrix.png')
        plt.close()

    def save(self):
        joblib.dump(self.mlp, "models/mlp_model.pkl")
        joblib.dump(self.tfidf, "models/tfidf.pkl")
        joblib.dump(self.le, "models/label_encoder.pkl")
        logger.info("Saved.")


if __name__ == "__main__":
    TRAIN_PATH = "../../data_processing/train_cleaned.csv"
    TEST_PATH = "../../data_processing/test_cleaned.csv"

    clf = MusicGenreClassifier()
    train_data, test_data = clf.load_data(TRAIN_PATH, TEST_PATH)

    clf.train(train_data, test_data)
    clf.evaluate(test_data)
    clf.save()