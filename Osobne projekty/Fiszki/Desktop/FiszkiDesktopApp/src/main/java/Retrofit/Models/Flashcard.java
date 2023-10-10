package Retrofit.Models;

import com.google.gson.annotations.SerializedName;

/**
 * Model class representing a Flashcard
 */
public class Flashcard {
    private String word, translatedWord, example, translatedExample, content, _id;
    private int countID;

    @SerializedName("body")
    private String text;

    /**
     * Constructs a FlashcardID object
     * @param word              the word
     * @param translatedWord    the translated word
     * @param example           the example sentence
     * @param translatedExample the translated example sentence
     * @param countID           the ID count
     * @param _id               the ID
     */
    public Flashcard(String word, String translatedWord, String example, String translatedExample, int countID, String _id){
        this.word = word;
        this.translatedWord = translatedWord;
        this.example = example;
        this.translatedExample = translatedExample;
        this._id = _id;
    }


    /**
     * Returns the word
     * @return the word
     */
    public String getWord() {
        return word;
    }

    /**
     * Returns the translated word
     * @return the translated word
     */
    public String getTranslatedWord() {
        return translatedWord;
    }

    /**
     * Returns the example sentence
     * @return the example sentence
     */
    public String getExample() {
        return example;
    }

    /**
     * Returns the translated example sentence
     * @return the translated example sentence
     */
    public String getTranslatedExample() {
        return translatedExample;
    }

    /**
     * Returns the content
     * @return the content
     */
    public String getContent() {
        return content;
    }

    /**
     * Returns the text
     * @return the text
     */
    public String getText() {
        return text;
    }

    /**
     * Returns the ID
     * @return the ID
     */
    public String get_id() {
        return _id;
    }
}
