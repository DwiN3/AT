package Controllers;

import java.io.IOException;

import Other.DateInstance;
import Retrofit.Models.User;
import app.App;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.PasswordField;
import javafx.scene.control.TextField;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import Retrofit.JsonPlaceholderAPI.JsonUser;

/**
 * This class is the controller for the first screen
 * This is the first screen of the application where the user can log in or go to the password change or registration section
 */
public class ControllerFirstScreen {
    @FXML
    private Label info_first;
    @FXML
    private TextField name_first;
    @FXML
    private PasswordField password_first;
    @FXML
    private Button login_button_first, register_button_first, reset_button_first, remind_button_first;
    @FXML
    private void switchActivity(String activity) throws IOException { App.setRoot(activity); }
    private DateInstance dateInstance = DateInstance.getInstance();

    /**
     * Initializes the controller
     */
    public void initialize(){
        reset_button_first.setStyle("-fx-background-color: transparent;");
        login_button_first.setOnAction(event -> {
            blockButtons(true);
            info_first.setStyle("-fx-text-fill: #00FF00;");
            info_first.setText("Trwa logowanie");
            info_first.setVisible(true);
            loginAccountRetrofit();
        });
        register_button_first.setOnAction(event -> {
            try {
                switchActivity("activity_register");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        remind_button_first.setOnAction(event -> {
            try {
                switchActivity("activity_login_reminder");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        reset_button_first.setOnAction(event -> {
            try {
                switchActivity("activity_reset_password");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * Disables or enables the buttons based on the specified isLoading value
     * @param isLoading true to disable the buttons and show loading state, false otherwise
     */
    public void blockButtons(boolean isLoading){
        double buttonOpacity = isLoading ? 1.0 : 1.0;
        login_button_first.setDisable(isLoading);
        login_button_first.setOpacity(buttonOpacity);
        register_button_first.setDisable(isLoading);
        register_button_first.setOpacity(buttonOpacity);
        remind_button_first.setDisable(isLoading);
        remind_button_first.setOpacity(buttonOpacity);
        reset_button_first.setDisable(isLoading);
        reset_button_first.setOpacity(buttonOpacity);
    }

    /**
     * Performs login using Retrofit
     */
    public void loginAccountRetrofit() {
        String loginString = String.valueOf(name_first.getText());
        String passwordString= String.valueOf(password_first.getText());
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://flashcard-app-api-bkrv.onrender.com/api/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonUser jsonUser = retrofit.create(JsonUser.class);
        User post = new User(loginString, passwordString);
        Call<User> call = jsonUser.login(post);

        call.enqueue(new Callback<User>() {
            @Override
            public void onResponse(Call<User> call, Response<User> response) {
                if(response.code() == 200){
                    Platform.runLater(() -> {
                        User post = response.body();
                        String TokenFromRetrofit = post.getToken();
                        dateInstance.setToken(TokenFromRetrofit);
                        dateInstance.setUserName(name_first.getText().toString());
                    });
                    blockButtons(false);
                    try {
                        switchActivity("activity_main_menu");
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                } else{
                    Platform.runLater(() -> {
                        info_first.setStyle("-fx-text-fill: #FF0000;");
                        info_first.setText("Błędne dane");
                        info_first.setVisible(true);
                        blockButtons(false);
                    });
                }
            }

            @Override
            public void onFailure(Call<User> call, Throwable t) {
                if(t.getMessage().equals("timeout")){
                    Platform.runLater(() -> {
                        info_first.setStyle("-fx-text-fill: #FF0000;");
                        info_first.setText("Uruchamianie serwera");
                        info_first.setVisible(true);
                        blockButtons(false);
                    });
                }
            }
        });
    }
}
