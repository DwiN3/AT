package com.example.Fiszki.flashcards.flashcard;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

/**
 * Repository interface for interacting with the database for Flashcard entities.
 */
public interface FlashcardRepository extends JpaRepository<Flashcard, Integer> {

    /**
     * Checks if a flashcard with the specified word exists.
     *
     * @param word The word to check for.
     * @return True if a flashcard with the given word exists, otherwise false.
     */
    boolean existsByWord(String word);

    /**
     * Checks if a flashcard with the specified translated word exists.
     *
     * @param translatedWord The translated word to check for.
     * @return True if a flashcard with the given translated word exists, otherwise false.
     */
    boolean existsByTranslatedWord(String translatedWord);

    /**
     * Finds flashcards by category.
     *
     * @param category The category of the flashcards.
     * @return A list of flashcards in the specified category.
     */
    List<Flashcard> findByCategory(String category);

    /**
     * Finds flashcards by author.
     *
     * @param author The author of the flashcards.
     * @return A list of flashcards created by the specified author.
     */
    List<Flashcard> findByAuthor(String author);

    /**
     * Finds flashcards by collection name and author.
     *
     * @param collectionName The name of the flashcard collection.
     * @param author         The author of the flashcards.
     * @return A list of flashcards in the specified collection and created by the specified author.
     */
    List<Flashcard> findByCollection_CollectionNameAndAuthor(String collectionName, String author);

    /**
     * Finds flashcards by collection.
     *
     * @param collection The flashcard collection.
     * @return A list of flashcards in the specified collection.
     */
    List<Flashcard> findByCollection(FlashcardCollection collection);

    /**
     * Checks if a flashcard with the specified word and collection name exists.
     *
     * @param word           The word to check for.
     * @param collectionName The name of the flashcard collection.
     * @return True if a flashcard with the given word and collection name exists, otherwise false.
     */
    boolean existsByWordAndCollection_CollectionName(String word, String collectionName);

    /**
     * Checks if a flashcard with the specified translated word and collection name exists.
     *
     * @param translatedWord The translated word to check for.
     * @param collectionName The name of the flashcard collection.
     * @return True if a flashcard with the given translated word and collection name exists, otherwise false.
     */
    boolean existsByTranslatedWordAndCollection_CollectionName(String translatedWord, String collectionName);
}