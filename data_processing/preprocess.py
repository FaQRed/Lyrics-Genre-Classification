import pandas as pd
import re


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    text = " ".join(text.split()).lower()
    return text


def process_existing_files():
    TRAIN_PATH = "../lyrics_data/train.csv"
    TEST_PATH = "../lyrics_data/test.csv"


    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)


    train_df['clean_text'] = train_df['Lyrics'].apply(clean_text)
    test_df['clean_text'] = test_df['Lyrics'].apply(clean_text)


    train_df['word_count'] = train_df['clean_text'].apply(lambda x: len(x.split()))
    test_df['word_count'] = test_df['clean_text'].apply(lambda x: len(x.split()))

    train_df = train_df[train_df['word_count'] >= 100].copy()
    test_df = test_df[test_df['word_count'] >= 100].copy()

    print(f"Final Train size: {len(train_df)}")
    print(f"Final Test size: {len(test_df)}")


    train_df.to_csv("train_cleaned.csv", index=False)
    test_df.to_csv("test_cleaned.csv", index=False)
    print("Cleaned files saved as 'train_cleaned.csv' and 'test_cleaned.csv'")


if __name__ == "__main__":
    process_existing_files()