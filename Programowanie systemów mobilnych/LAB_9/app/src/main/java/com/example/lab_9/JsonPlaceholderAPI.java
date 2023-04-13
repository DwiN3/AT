package com.example.lab_9;

import java.util.List;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.PATCH;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Path;

public interface JsonPlaceholderAPI {
    @GET("messages")
    Call<List<Post>> getPostsAll();

    @GET("messages?last=8")
    Call<List<Post>> getPostsLast8();

    @POST("message")
    Call<List<Post>> createMessage(@Body Post post);

    @PUT("message/{id}")
    Call<List<Post>> putPost(@Path("id") String id, @Body Post post);

    @PATCH("message/{id}")
    Call<List<Post>> patchPost(@Path("id") String id, @Body Post post);

    @DELETE("message/{id}")
    Call<List<Post>> deletePost(@Path("id") String id);
}