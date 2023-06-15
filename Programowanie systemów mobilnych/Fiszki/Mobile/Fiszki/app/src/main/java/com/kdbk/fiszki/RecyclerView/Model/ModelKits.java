package com.kdbk.fiszki.RecyclerView.Model;

import com.kdbk.fiszki.Retrofit.Models.FlashcardID;
import java.util.ArrayList;

public class ModelKits {
    private int numberOfCards, gamesPlayed,countID;
    private String nameKit, TEXT, wordID;
    private ArrayList<FlashcardID> list = new ArrayList<FlashcardID>();

    public ModelKits(String nameKit, String TEXT, int numberOfCards, int countID, int gamesPlayed, String wordID){
        this.numberOfCards = numberOfCards;
        this.nameKit = nameKit;
        this.TEXT = TEXT;
        this.countID = countID;
        this.gamesPlayed = gamesPlayed;
        this.wordID = wordID;
    }

    public int getNumberOfCards() {
        return numberOfCards;
    }

    public String getNameKit() {
        return nameKit;
    }

    public String getTEXT() {
        return TEXT;
    }

    public int getCountID() {
        return countID;
    }
    public int getGamesPlayed() {
        return gamesPlayed;
    }

    public String getWordID() {
        return wordID;
    }

    public ArrayList<FlashcardID> getList() {
        return list;
    }

    public void setList(ArrayList<FlashcardID> list) {
        this.list = list;
    }

}
