package com.example.Fiszki.flashcards.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a request for limiting the number of flashcards retrieved based on category.
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FlashcardCategoryLimitRequest {

    /**
     * The maximum number of flashcards to be retrieved for a specific category.
     */
    private int limit;
}