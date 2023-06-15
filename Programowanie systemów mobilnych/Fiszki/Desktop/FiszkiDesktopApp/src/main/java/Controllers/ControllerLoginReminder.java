package Controllers;

import Mail.SendEmailTLS;
import Retrofit.JsonPlaceholderAPI.JsonUser;
import Retrofit.Models.User;
import app.App;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import java.io.IOException;

/**
 * This class is the controller for the reminder password screen
 * This class handles the view logic responsible for remembering the user's password
 */
public class ControllerLoginReminder {
    @FXML
    private Label info_login_reminder;
    @FXML
    private TextField email_reminder;
    @FXML
    private Button button_reminder, back_button_reminder;

    @FXML
    private void switchActivity(String activity) throws IOException {
        App.setRoot(activity);
    }

    /**
     * Initializes the controller
     */
    public void initialize() {
        button_reminder.setOnAction(event -> {
            info_login_reminder.setVisible(false);


            if (!email_reminder.getText().isEmpty()) {
                Platform.runLater(() -> {
                    reminderLoginRetrofit();
                    blockButtons(true);
                });
            }
        });


        back_button_reminder.setOnAction(event -> {
            try {
                switchActivity("activity_first_screen");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * Disables or enables the buttons based on the specified isLoading value
     *
     * @param isLoading true to disable the buttons and show loading state, false otherwise
     */
    public void blockButtons(boolean isLoading) {
        double buttonOpacity = isLoading ? 1.0 : 1.0;
        button_reminder.setDisable(isLoading);
        button_reminder.setOpacity(buttonOpacity);
        back_button_reminder.setDisable(isLoading);
        back_button_reminder.setOpacity(buttonOpacity);
    }

    /**
     * Performs a user password reminder using Retrofit
     */
    public void reminderLoginRetrofit() {
        String emailString = String.valueOf(email_reminder.getText());
        String subject = "Przypomnienie loginu w aplikacji fiszki";

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://flashcard-app-api-bkrv.onrender.com/api/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonUser jsonUser = retrofit.create(JsonUser.class);
        User post = new User(emailString);
        Call<String> call = jsonUser.getLogin(post);

        call.enqueue(new Callback<String>() {
            @Override
            public void onResponse(Call<String> call, Response<String> response) {
                int statusCode = response.code();
                if (statusCode == 200) {
                    String nick = response.body();
                    if (nick != null && !nick.isEmpty()) {
                        String message = "Nazwa: " + nick;
                        SendEmailTLS sendEmailTLS;
                        try {
                            sendEmailTLS = new SendEmailTLS(emailString, subject, message);
                            sendEmailTLS.sendMessage();
                            Platform.runLater(() -> {
                                info_login_reminder.setStyle("-fx-text-fill: #00FF00;");
                                info_login_reminder.setText("Login został wysłany na maila");
                                info_login_reminder.setVisible(true);
                                blockButtons(false);
                            });
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    }
                } else {
                    Platform.runLater(() -> {
                        info_login_reminder.setStyle("-fx-text-fill: #FF0000;");
                        info_login_reminder.setText("Błędny mail");
                        info_login_reminder.setVisible(true);
                        blockButtons(false);
                    });
                }
            }

            @Override
            public void onFailure(Call<String> call, Throwable t) {
                if (t.getMessage().equals("timeout")) {
                    Platform.runLater(() -> {
                        info_login_reminder.setStyle("-fx-text-fill: #FF0000;");
                        info_login_reminder.setText("Uruchamianie serwera");
                        info_login_reminder.setVisible(true);
                        blockButtons(false);
                    });
                }
            }
        });
    }
}