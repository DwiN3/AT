package com.example.Fiszki.flashcards.response;

import lombok.*;
import java.util.List;

/**
 * Represents a response containing flashcard collection information, including the collection name and associated flashcards.
 */
@Getter
@Builder
public class FlashcardCollectionResponse {

    /**
     * The name of the flashcard collection.
     */
    private String name_kit;

    /**
     * The list of flashcards associated with the flashcard collection.
     */
    private List<FlashcardReturnResponse> flashcards;
}