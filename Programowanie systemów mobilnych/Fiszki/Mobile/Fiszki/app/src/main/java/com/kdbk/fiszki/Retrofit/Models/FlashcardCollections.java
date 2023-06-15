package com.kdbk.fiszki.Retrofit.Models;

public class FlashcardCollections {
        private String collectionName, _id;
        private String [] flashcards;

        public FlashcardCollections() {
        }

        public FlashcardCollections(String collectionName, String _id, String [] flashcards) {
            this.collectionName = collectionName;
            this._id = _id;
            this.flashcards = flashcards;
        }

        public String getCollectionName() {
            return collectionName;
        }
        public String getId() {
            return _id;
        }
        public int getFlashcardsSize() {
            return flashcards.length;
        }
        public String getPointsAvaible() {
        return String.valueOf(flashcards.length * 10);
    }
}
