package Retrofit.JsonPlaceholderAPI;

import Retrofit.Models.Flashcard;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Path;
import java.util.List;

/**
 * Interface representing the JSON Flashcards Collections API
 */
public interface JsonFlashcardsCollections {

    /**
     * Retrieves the flashcards from a specific category
     * @param collectionName the name of the flashcards collection category
     * @return a Call object containing a list of lists of FlashcardID objects
     */
    @GET("flashcards-collections/category/{collectionName}")
    Call<List<List<Flashcard>>> getCategory(@Path("collectionName") String collectionName);
}
