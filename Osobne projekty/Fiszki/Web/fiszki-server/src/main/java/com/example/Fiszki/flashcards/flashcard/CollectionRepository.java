package com.example.Fiszki.flashcards.flashcard;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional;

/**
 * Repository interface for interacting with the database for FlashcardCollection entities.
 */
public interface CollectionRepository extends JpaRepository<FlashcardCollection, Integer> {

    /**
     * Finds a flashcard collection by its name.
     *
     * @param collectionName The name of the flashcard collection.
     * @return An Optional containing the flashcard collection if found, otherwise empty.
     */
    Optional<FlashcardCollection> findByCollectionName(String collectionName);

    /**
     * Checks if a flashcard collection with the specified name and author exists.
     *
     * @param collectionName The name of the flashcard collection.
     * @param author         The author of the flashcard collection.
     * @return True if a flashcard collection with the given name and author exists, otherwise false.
     */
    boolean existsByCollectionNameAndAuthor(String collectionName, String author);

    /**
     * Finds flashcard collections by name and author.
     *
     * @param collectionName The name of the flashcard collection.
     * @param author         The author of the flashcard collection.
     * @return A list of flashcard collections matching the specified name and author.
     */
    List<FlashcardCollection> findByCollectionNameAndAuthor(String collectionName, String author);

    /**
     * Finds all flashcard collections by author.
     *
     * @param author The author of the flashcard collections.
     * @return A list of flashcard collections created by the specified author.
     */
    List<FlashcardCollection> findCollectionByAuthor(String author);
}