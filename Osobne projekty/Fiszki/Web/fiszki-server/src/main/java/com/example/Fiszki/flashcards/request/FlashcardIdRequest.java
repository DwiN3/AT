package com.example.Fiszki.flashcards.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a request for a flashcard identified by its unique ID.
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FlashcardIdRequest {

    /**
     * The unique identifier of the flashcard.
     */
    private Integer id;
}