package com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI;

import com.kdbk.fiszki.Retrofit.Models.FlashcardCollections;
import com.kdbk.fiszki.Retrofit.Models.FlashcardCollectionsWords;
import com.kdbk.fiszki.Retrofit.Models.FlashcardID;

import java.util.ArrayList;
import java.util.List;
import retrofit2.Call;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.Path;

public interface JsonFlashcardsCollections {
    @GET("flashcards-collections")
    Call <List<FlashcardCollections>> getAllFlashcardsCollections();
    @GET("flashcards-collections/{collectionName}")
    Call<FlashcardCollectionsWords> getKit(@Path("collectionName") String collectionName);
    @GET("flashcards-collections/category/{collectionName}")
    Call<List<List<FlashcardID>>> getCategory(@Path("collectionName") String collectionName);
    @DELETE("flashcards-collections/{collectionName}")
    Call<FlashcardCollections> deleteFlashcardsCollections(@Path("collectionName") String collectionName);
}
