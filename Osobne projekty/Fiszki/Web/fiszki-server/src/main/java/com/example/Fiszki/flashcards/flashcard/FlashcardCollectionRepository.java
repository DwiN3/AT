package com.example.Fiszki.flashcards.flashcard;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

/**
 * Repository interface for interacting with the database for FlashcardCollection entities.
 */
public interface FlashcardCollectionRepository extends JpaRepository<FlashcardCollection, Integer> {

    /**
     * Finds a flashcard collection by its name.
     *
     * @param collectionName The name of the flashcard collection.
     * @return An Optional containing the flashcard collection if found, otherwise empty.
     */
    Optional<FlashcardCollection> findByCollectionName(String collectionName);

    /**
     * Checks if a flashcard collection with the specified name exists.
     *
     * @param collectionName The name of the flashcard collection.
     * @return True if a flashcard collection with the given name exists, otherwise false.
     */
    boolean existsByCollectionName(String collectionName);
}