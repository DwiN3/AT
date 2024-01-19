package com.example.Fiszki.flashcards.flashcard;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a flashcard collection entity with information such as ID, collection name, and author.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "collections")
public class FlashcardCollection {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    /**
     * The name of the flashcard collection.
     */
    private String collectionName;

    /**
     * The author of the flashcard collection.
     */
    private String author;

    /**
     * Gets the ID of the flashcard collection.
     *
     * @return The ID of the flashcard collection.
     */
    public Integer getId() {
        return id;
    }

    /**
     * Gets the author of the flashcard collection.
     *
     * @return The author of the flashcard collection.
     */
    public String getAuthor() {
        return author;
    }

    /**
     * Sets the author of the flashcard collection.
     *
     * @param author The author of the flashcard collection.
     */
    public void setAuthor(String author) {
        this.author = author;
    }

    /**
     * Sets the ID of the flashcard collection.
     *
     * @param id The ID of the flashcard collection.
     */
    public void setId(Integer id) {
        this.id = id;
    }

    /**
     * Gets the name of the flashcard collection.
     *
     * @return The name of the flashcard collection.
     */
    public String getCollectionName() {
        return collectionName;
    }

    /**
     * Sets the name of the flashcard collection.
     *
     * @param collectionName The name of the flashcard collection.
     */
    public void setCollectionName(String collectionName) {
        this.collectionName = collectionName;
    }
}