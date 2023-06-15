package com.kdbk.fiszki.Retrofit.Models;

import com.google.gson.annotations.SerializedName;

public class FlashcardID {
    private String word, translatedWord, example, translatedExample, content, _id;

    @SerializedName("body")
    private String text;

    public FlashcardID() {

    }

    public FlashcardID(String word, String translatedWord, String example, String translatedExample) {
        this.word = word;
        this.translatedWord = translatedWord;
        this.example = example;
        this.translatedExample = translatedExample;
    }

    public FlashcardID(String _id, String word, String translatedWord, String example, String translatedExample) {
        this.word = word;
        this.translatedWord = translatedWord;
        this.example = example;
        this.translatedExample = translatedExample;
        this._id = _id;
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
    public String get_id() {
        return _id;
    }
}
