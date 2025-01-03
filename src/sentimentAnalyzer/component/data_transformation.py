import re
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentimentAnalyzer.logging import logger
from sentimentAnalyzer.config.configuration import DataTransformationConfig
from sentimentAnalyzer.utils.common import (DataInfo,
                                            read_dataset,
                                            save_transformed_data_file)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def lower_case_converter(self, text):
        return text.lower()

    def basic_remove_process(self, text):
        # Remove users
        no_user_text = re.sub(r'@\w+', '', text)
        # Remove Numbers
        remove_number_text = re.sub(r'\d+', '', no_user_text)
        # Remove Punctuation
        translator = str.maketrans('', '', string.punctuation)
        remove_punc_text = remove_number_text.translate(translator)
        # Remove URLs
        remove_url_text = re.sub(r'http\S+|www\S+', '', remove_punc_text)
        # Remove Extra Space
        final_text = " ".join(remove_url_text.split())

        return final_text

    def text_tokenizer(self, text):
        tokenize_lst_text = word_tokenize(text)
        return tokenize_lst_text

    def stop_word_removal(self, text_lst):
        STOPWORD =  set(stopwords.words("english"))
        stopword_remove_text = [word for word in text_lst if word not in STOPWORD]
        return stopword_remove_text

    def text_stemmer(self, text_lst):
        stemmer = SnowballStemmer('english')
        stem_text = [stemmer.stem(word) for word in text_lst]
        return stem_text
    
    def text_transformer(self, path: Path, data_info: DataInfo):
        df = read_dataset(path, encoding=self.config.encoder)
        df = df[["target", "text"]]

        df['target'] = df['target'].replace(4, 1)

        df['text'] = df['text'].apply(lambda x: self.lower_case_converter(x))
        df['text'] = df['text'].apply(lambda x: self.basic_remove_process(x))
        df['text'] = df['text'].apply(lambda x: self.text_tokenizer(x))
        df['text'] = df['text'].apply(lambda x: self.stop_word_removal(x))
        df['text'] = df['text'].apply(lambda x: self.text_stemmer(x))

        df['text'] = df['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        logger.info(f"{data_info.value} has been Pre-Processed")

        return df['text'], df['target']
    

    def data_transformation(self) -> None:
        X_train, y_train = self.text_transformer(path=self.config.train_data_file, data_info=DataInfo.TRAINING)
        X_test, y_test = self.text_transformer(path=self.config.test_data_file, data_info=DataInfo.TESTING)
        
        vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=300000)
        vectoriser.fit(X_train)
        X_train = vectoriser.transform(X_train)
        X_test = vectoriser.transform(X_test)

        # Saving the Data
        # Training Data
        x_tr_path = Path(f"{self.config.train_transformed_dir}/X_train.npz")
        y_tr_path = Path(f"{self.config.train_transformed_dir}/y_train.npy")
        save_transformed_data_file(x_tr_path, X_train, data_info=DataInfo.TRAINING)
        save_transformed_data_file(y_tr_path, y_train, data_info=DataInfo.TRAINING)
        # Testing Day
        x_test_path = Path(f"{self.config.test_transformed_dir}/X_test.npz")
        y_test_path = Path(f"{self.config.test_transformed_dir}/y_test.npy")
        save_transformed_data_file(x_test_path, X_test, data_info=DataInfo.TESTING)
        save_transformed_data_file(y_test_path, y_test, data_info=DataInfo.TESTING)
        return None
    