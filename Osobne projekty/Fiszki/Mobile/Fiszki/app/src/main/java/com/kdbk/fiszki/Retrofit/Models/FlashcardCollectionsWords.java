package com.kdbk.fiszki.Retrofit.Models;

import java.util.ArrayList;

public class FlashcardCollectionsWords {
    private String collectionName, _id;
    private ArrayList<FlashcardID> flashcards;

    public FlashcardCollectionsWords() {
    }

    public FlashcardCollectionsWords(String collectionName, String _id) {
        this.collectionName = collectionName;
        this._id = _id;
    }

    public String getCollectionName() {
        return collectionName;
    }

    public String getId() {
        return _id;
    }

        public ArrayList<FlashcardID> getFlashcards() {
            return flashcards;
        }

        public void setFlashcards(ArrayList<FlashcardID> flashcards) {
            this.flashcards = flashcards;
        }
}
