import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_and_extract():
    dataset = "mateibejan/multilingual-lyrics-for-genre-classification"
    path = "../lyrics_data"


    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Downloading {dataset}...")
        api.dataset_download_files(dataset, path=path, unzip=True)
        print("Done!")
    else:
        print("Dataset directory already exists.")


if __name__ == "__main__":
    download_and_extract()