package com.example.Fiszki.flashcards.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a response containing information about a flashcard, including its ID, author, category, words, and examples.
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FlashcardReturnResponse {

    /**
     * The unique identifier of the flashcard.
     */
    private Integer id;

    /**
     * The author of the flashcard.
     */
    private String author;

    /**
     * The category of the flashcard.
     */
    private String category;

    /**
     * The word on the flashcard.
     */
    private String word;

    /**
     * The translated word on the flashcard.
     */
    private String translatedWord;

    /**
     * An example sentence using the word on the flashcard.
     */
    private String example;

    /**
     * The translated example sentence.
     */
    private String translatedExample;
}