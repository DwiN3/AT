package com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI;

import com.kdbk.fiszki.Retrofit.Models.FlashcardID;
import com.kdbk.fiszki.Retrofit.Models.Flashcards;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Path;

public interface JsonFlashcards {
    @POST("flashcards/{collectionName}")
    Call<Flashcards> addFlashcard(@Path("collectionName") String collectionName, @Body Flashcards flashcards);
    @DELETE("flashcards/{flashcardsId}")
    Call<FlashcardID> deleteFlashcards(@Path("flashcardsId") String flashcardsId);
    @PUT("flashcards/{flashcardsId}")
    Call<FlashcardID> editFlashcards(@Path("flashcardsId") String flashcardsId, @Body FlashcardID flashCardsID);
    @GET("flashcards/{collectionName}")
    Call<FlashcardID> getFlashcard(@Path("collectionName") String flashcardsId);
}
