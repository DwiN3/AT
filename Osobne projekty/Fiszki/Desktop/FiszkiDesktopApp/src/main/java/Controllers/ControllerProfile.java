package Controllers;

import Other.DateInstance;
import Retrofit.JsonPlaceholderAPI.JsonUser;
import Retrofit.Models.UserLVL;
import app.App;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import java.io.IOException;

/**
 * This class is the controller for the profile screen
 * The screen displays information about the username, the current level and the number of points to the next level
 */
public class ControllerProfile {
    @FXML
    private Label nick_user_profile,lvl_profile,points_to_next_LVL_profile;
    @FXML
    private Button back_to_menu_button_profile;
    @FXML
    private void switchActivity(String activity) throws IOException { App.setRoot(activity); }
    private DateInstance dateInstance = DateInstance.getInstance();

    /**
     * Initializes the controller
     */
    public void initialize(){
        getInfoUserRetrofit();

        back_to_menu_button_profile.setOnAction(event -> {
            try {
                switchActivity("activity_main_menu");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * Retrieving user data using retrofit
     */
    public void getInfoUserRetrofit() {
        OkHttpClient client = new OkHttpClient.Builder().addInterceptor(new Interceptor() {
            @Override
            public okhttp3.Response intercept(Chain chain) throws IOException {
                Request newRequest = chain.request().newBuilder()
                        .addHeader("Authorization", "Bearer " + dateInstance.getToken())
                        .build();
                return chain.proceed(newRequest);
            }
        }).build();

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://flashcard-app-api-bkrv.onrender.com/api/")
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonUser jsonUser = retrofit.create(JsonUser.class);
        Call<UserLVL> call = jsonUser.getUserLVL();

        call.enqueue(new Callback<UserLVL>() {
            @Override
            public void onResponse(Call<UserLVL> call, Response<UserLVL> response) {
                if (response.isSuccessful() && response.body() != null) {
                    Platform.runLater(() -> {
                        nick_user_profile.setText("Login: " + dateInstance.getUserName());
                        lvl_profile.setText("Poziom: " + response.body().getLevel() + " lvl");
                        points_to_next_LVL_profile.setText("Następny poziom: " + response.body().getPoints() + "/" + response.body().getRequiredPoints() + " pkt");
                    });
                }
            }

            @Override
            public void onFailure(Call<UserLVL> call, Throwable t) { }
        });
    }
}




