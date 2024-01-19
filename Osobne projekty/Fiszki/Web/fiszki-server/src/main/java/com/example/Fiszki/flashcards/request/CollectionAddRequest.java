package com.example.Fiszki.flashcards.request;

import lombok.Data;

/**
 * Represents a request for adding a new flashcard collection.
 */
@Data
public class CollectionAddRequest {

    /**
     * The name of the flashcard collection to be added.
     */
    private String collectionName;
}