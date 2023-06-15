package Controllers;

import java.io.IOException;
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
 * This class is the controller for the register screen
 * This section allows you to create a user account
 */
public class ControllerRegister {
    @FXML
    private Label info_register;
    @FXML
    private TextField name_register, email_register;
    @FXML
    private PasswordField password_register, password_re_register;
    @FXML
    private Button register_button_register, back_button_register;
    @FXML
    private void switchActivity(String activity) throws IOException { App.setRoot(activity); }

    /**
     * Initializes the controller
     */
    public void initialize(){
        register_button_register.setOnAction(event -> {
            blockButtons(true);
            info_register.setVisible(false);
            registerAccountRetrofit();
        });

        back_button_register.setOnAction(event -> {
            try {
                switchActivity("activity_first_screen");
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
        register_button_register.setDisable(isLoading);
        register_button_register.setOpacity(buttonOpacity);
        back_button_register.setDisable(isLoading);
        back_button_register.setOpacity(buttonOpacity);
    }

    /**
     * Performs user registrations through retrofit
     */
    public void registerAccountRetrofit(){
        String loginString = String.valueOf(name_register.getText());
        String emailString = String.valueOf(email_register.getText());
        String passwordString= String.valueOf(password_register.getText());
        String passwordReString= String.valueOf(password_re_register.getText());

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://flashcard-app-api-bkrv.onrender.com/api/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonUser jsonUser = retrofit.create(JsonUser.class);
        User post = new User(emailString, passwordString, passwordReString, loginString);
        Call<User> call = jsonUser.register(post);

        call.enqueue(new Callback<User>() {
            @Override
            public void onResponse(Call<User> call, Response<User> response) {
                if(!response.isSuccessful()){
                    Platform.runLater(() -> {
                        info_register.setStyle("-fx-text-fill: #FF0000;");
                        info_register.setText("Błędne dane");
                        info_register.setVisible(true);
                        blockButtons(false);
                    });
                }
            }

            @Override
            public void onFailure(Call<User> call, Throwable t) {
                if(t.getMessage().equals("timeout")){
                    Platform.runLater(() -> {
                        info_register.setStyle("-fx-text-fill: #FF0000;");
                        info_register.setText("Uruchamianie serwera");
                        info_register.setVisible(true);
                        blockButtons(false);
                    });
                }else{
                    Platform.runLater(() -> {
                        info_register.setStyle("-fx-text-fill: #00FF00;");
                        info_register.setText("Utworzono konto pomyślnie");
                        info_register.setVisible(true);
                        register_button_register.setVisible(false);
                        blockButtons(false);
                    });
                }
            }
        });
    }
}