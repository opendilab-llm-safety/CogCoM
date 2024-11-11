# cogcom/data/download_nltk_data.py
import nltk

def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading averaged_perceptron_tagger...")
        nltk.download('averaged_perceptron_tagger')

if __name__ == "__main__":
    print("开始下载NLTK资源...")
    download_nltk_resources()
    print("NLTK资源下载完成。")