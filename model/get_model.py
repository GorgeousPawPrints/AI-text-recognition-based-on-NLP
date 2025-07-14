import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump


if __name__ == '__main__':
    df = pd.read_csv('../data_set/preprocessed_AI_Human.csv')
    y = df['generated']
    X = df['text']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer()),  # Step 1: CountVectorizer
        ('tfidf_transformer', TfidfTransformer()),  # Step 2: TF-IDF Transformer
        ('naive_bayes', MultinomialNB())])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))

    # 保存整个Pipeline到文件（包含所有预处理步骤）
    dump(pipeline, 'text_classification_model.pkl')

    # 加载模型的代码示例
    # loaded_model = load('text_classification_model.pkl')