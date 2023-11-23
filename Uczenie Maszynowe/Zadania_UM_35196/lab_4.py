import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Laboratorium 4

# 1. Wczytaj dane z plików
# Wczytaj dane z pliku positive.txt
with open('positive.txt', 'r', encoding='utf-8') as positive_file:
    positive_data = positive_file.readlines()
    # 2. Przydziel klasę '0' (positive)
    positive_labels = [0] * len(positive_data)

# Wczytaj dane z pliku negative.txt
with open('negative.txt', 'r', encoding='utf-8') as negative_file:
    negative_data = negative_file.readlines()
    # 2. Przydziel klasę '1' (negative)
    negative_labels = [1] * len(negative_data)

# 3. Utwórz dataframe z nagłówkami „text” i „class”
df_positive = pd.DataFrame({'text': positive_data, 'class': positive_labels})
df_negative = pd.DataFrame({'text': negative_data, 'class': negative_labels})
# Połącz ramki danych
df = pd.concat([df_positive, df_negative], ignore_index=True)

# 4. Zmieszaj (shuffle) zestaw danych
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Przedstaw dane za pomocą df.head
print("Pierwsze 5 wierszy zestawu danych po przetasowaniu:")
print(df.head())

# 6. Utwórz słownik „Bag of Words”
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(df['text'])

# 7. Pokaż, które słowa występują najczęściej dla całego zbioru
word_freq = dict(zip(vectorizer.get_feature_names_out(), bag_of_words.sum(axis=0).tolist()[0]))
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
print("\nNajczęściej występujące słowa dla całego zbioru:")
print(sorted_word_freq[:10])

# 8. Pokaż, które słowa występują najczęściej dla zbioru negative a które dla positive
negative_subset = df[df['class'] == 1]
positive_subset = df[df['class'] == 0]

negative_bag_of_words = vectorizer.transform(negative_subset['text'])
positive_bag_of_words = vectorizer.transform(positive_subset['text'])

negative_word_freq = dict(zip(vectorizer.get_feature_names_out(), negative_bag_of_words.sum(axis=0).tolist()[0]))
positive_word_freq = dict(zip(vectorizer.get_feature_names_out(), positive_bag_of_words.sum(axis=0).tolist()[0]))

sorted_negative_word_freq = sorted(negative_word_freq.items(), key=lambda x: x[1], reverse=True)
sorted_positive_word_freq = sorted(positive_word_freq.items(), key=lambda x: x[1], reverse=True)

print("\nNajczęściej występujące słowa dla zbioru negative:")
print(sorted_negative_word_freq[:10])

print("\nNajczęściej występujące słowa dla zbioru positive:")
print(sorted_positive_word_freq[:10])