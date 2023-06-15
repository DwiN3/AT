package com.kdbk.fiszki.Retrofit.Models;

import com.google.gson.annotations.SerializedName;

public class Flashcards {
    private String collectionName, language, category, word, translatedWord, example, translatedExample, content;

    @SerializedName("body")
    private String text;

    public Flashcards() {
    }

    public Flashcards(String collectionName, String language, String category, String word, String translatedWord, String example, String translatedExample) {
        this.collectionName = collectionName;
        this.language = language;
        this.category = category;
        this.word = word;
        this.translatedWord = translatedWord;
        this.example = example;
        this.translatedExample = translatedExample;
    }


    public String getCollectionName() {
        return collectionName;
    }

    public String getLanguage() {
        return language;
    }

    public String getCategory() {
        return category;
    }

    public String getWord() {
        return word;
    }

    public String getTranslatedWord() {
        return translatedWord;
    }

    public String getExample() {
        return example;
    }

    public String getTranslatedExample() {
        return translatedExample;
    }

    public String getContent() {
        return content;
    }

    public String getText() {
        return text;
    }

}
