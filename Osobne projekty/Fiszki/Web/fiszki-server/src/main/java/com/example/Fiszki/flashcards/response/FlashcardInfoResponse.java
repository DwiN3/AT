package com.example.Fiszki.flashcards.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a response containing information about a flashcard operation or request.
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FlashcardInfoResponse {

    /**
     * The response message containing information about a flashcard operation or request.
     */
    private String response;
}