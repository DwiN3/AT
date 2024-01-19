package com.example.Fiszki.flashcards.flashcard;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a flashcard entity with information such as author, collection, category, words, and examples.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "flashcards")
public class Flashcard {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY) // Auto generating id
    private Integer id;

    /**
     * The author of the flashcard.
     */
    private String author;

    /**
     * The flashcard collection to which this flashcard belongs.
     */
    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "collection_id")
    private FlashcardCollection collection;

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

    /**
     * Gets the flashcard collection to which this flashcard belongs.
     *
     * @return The flashcard collection.
     */
    public FlashcardCollection getCollection() {
        return collection;
    }

    /**
     * Sets the flashcard collection to which this flashcard belongs.
     *
     * @param collection The flashcard collection.
     */
    public void setCollection(FlashcardCollection collection) {
        this.collection = collection;
    }

    /**
     * Gets the category of the flashcard.
     *
     * @return The flashcard category.
     */
    public String getCategory() {
        return category;
    }

    /**
     * Gets the word on the flashcard.
     *
     * @return The word on the flashcard.
     */
    public String getWord() {
        return word;
    }

    /**
     * Gets the translated word on the flashcard.
     *
     * @return The translated word on the flashcard.
     */
    public String getTranslatedWord() {
        return translatedWord;
    }

    /**
     * Gets an example sentence using the word on the flashcard.
     *
     * @return The example sentence.
     */
    public String getExample() {
        return example;
    }

    /**
     * Gets the translated example sentence.
     *
     * @return The translated example sentence.
     */
    public String getTranslatedExample() {
        return translatedExample;
    }

    /**
     * Gets the author of the flashcard.
     *
     * @return The author of the flashcard.
     */
    public String getAuthor() {
        return author;
    }
}