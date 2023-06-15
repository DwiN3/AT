package com.kdbk.fiszki.RecyclerView.Model;

public class ModelShowKitsEdit {
    private String word, translateWord,sentens,sentensTranslate, wordID;
    private int countID;

    public ModelShowKitsEdit(String word, String translateWord, String sentens, String sentensTranslate, int countID, String wordID){
        this.word = word;
        this.translateWord = translateWord;
        this.sentens = sentens;
        this.sentensTranslate = sentensTranslate;
        this.wordID = wordID;
    }

    public int getCountID() {
        return countID;
    }
    public String getWord() {
        return word;
    }

    public String getTranslateWord() {
        return translateWord;
    }

    public String getSentens() {
        return sentens;
    }

    public String getSentensTranslate() {
        return sentensTranslate;
    }

    public String getWordID() {
        return wordID;
    }
}
