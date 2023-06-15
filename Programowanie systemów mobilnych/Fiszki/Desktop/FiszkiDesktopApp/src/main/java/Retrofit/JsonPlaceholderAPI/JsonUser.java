package Retrofit.JsonPlaceholderAPI;

import Retrofit.Models.User;
import Retrofit.Models.UserLVL;
import retrofit2.Call;
import retrofit2.http.*;

/**
 * Interface representing the JSON User API
 */
public interface JsonUser {
    /**
     * Sends a login request
     * @param user the User object containing login information
     * @return a Call object containing a Login object
     */
    @POST("users/login")
    Call<User> login(@Body User user);

    /**
     * Sends a register request
     * @param user the User object containing registration information
     * @return a Call object containing a Register object
     */
    @POST("users/sing-up")
    Call<User> register(@Body User user);

    /**
     * Sends a password reset request
     * @param user the User object containing password reset information
     * @return a Call object containing a Register object
     */
    @PUT("users/password-reset")
    Call<String> resetPassword(@Body User user);

    /**
     * Retrieves the user's password and login
     * @param user the User object containing login information
     * @return a Call object containing a Register object
     */
    @POST("users/nick-remind")
    Call<String> getLogin(@Body User user);

    /**
     * Sends a points request to update the user's level
    * @param userLVL        the UserLVL object containing level information
     * @return a Call object containing a UserLVL object
    */
    @PUT("users/users-level")
    Call<UserLVL> points(@Body UserLVL userLVL);

    /**
     * Retrieves the user's level
     * @return a Call object containing a UserLVL object
     */
    @GET("users/users-level")
    Call<UserLVL> getUserLVL();
}
