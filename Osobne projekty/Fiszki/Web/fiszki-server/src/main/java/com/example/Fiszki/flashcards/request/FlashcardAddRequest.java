package com.example.Fiszki.flashcards.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a request for adding a new flashcard.
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FlashcardAddRequest {

    /**
     * The name of the flashcard collection to which the new flashcard belongs.
     */
    private String collectionName;

    /**
     * The category of the new flashcard.
     */
    private String category;

    /**
     * The word on the new flashcard.
     */
    private String word;

    /**
     * The translated word on the new flashcard.
     */
    private String translatedWord;

    /**
     * An example sentence using the word on the new flashcard.
     */
    private String example;

    /**
     * The translated example sentence.
     */
    private String translatedExample;
}