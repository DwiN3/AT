package com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI;

import com.kdbk.fiszki.Retrofit.Models.Flashcards;
import com.kdbk.fiszki.Retrofit.Models.Login;
import com.kdbk.fiszki.Retrofit.Models.Register;
import com.kdbk.fiszki.Retrofit.Models.UserLVL;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Path;

public interface JsonUser {
    @POST("users/login")
    Call<Login> login(@Body Login login);
    @POST("users/sing-up")
    Call<Register> register(@Body Register register);
    @PUT("users/password-reset")
    Call<String> resetPassword(@Body Register register);
    @PUT("users/users-level")
    Call<UserLVL> points(@Body UserLVL userLVL);
    @GET("users/users-level")
    Call<UserLVL> getUserLVL();
}
