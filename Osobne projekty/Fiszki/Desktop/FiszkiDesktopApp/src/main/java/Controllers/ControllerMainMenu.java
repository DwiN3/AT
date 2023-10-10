package Controllers;

import app.App;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import Other.DateInstance;

import java.io.IOException;

/**
 * This class is the controller for the main menu screen
 * It handles user interactions and manages the behavior of the main menu UI elements
 */
public class ControllerMainMenu {
    @FXML
    private Label nick_user_menu;
    @FXML
    private ChoiceBox<String> category_choice_box_menu;
    @FXML
    private Button profile_button_menu, game_quiz_button_menu, game_wpis_button_menu, log_out_button_menu;
    @FXML
    private void switchActivity(String activity) throws IOException { App.setRoot(activity); }
    private DateInstance dateInstance = DateInstance.getInstance();
    private String selectedCategory;

    /**
     * Initializes the controller
     */
    public void initialize(){
        nick_user_menu.setText("Witaj "+ dateInstance.getUserName());
        selectedCategory = "zwierzeta";
        dateInstance.setCategoryName(selectedCategory);

        profile_button_menu.setOnAction(event -> {
            try {
                switchActivity("activity_profile");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        game_quiz_button_menu.setOnAction(event -> {
            try {
                dateInstance.setGameMode("quiz");
                switchActivity("activity_quiz_game");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        game_wpis_button_menu.setOnAction(event -> {
            try {
                dateInstance.setGameMode(("wpis"));
                switchActivity("activity_wpis_game");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        log_out_button_menu.setOnAction(event -> {
            dateInstance.setToken("");
            dateInstance.setUserName("");
            try {
                switchActivity("activity_first_screen");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        category_choice_box_menu.getItems().addAll("zwierzeta","dom", "zakupy", "praca", "zdrowie", "czlowiek", "turystyka","jedzenie","edukacja", "inne");
        category_choice_box_menu.setValue("zwierzeta");
        selectedCategory = "zwierzeta";

        category_choice_box_menu.setOnAction(event -> {
            selectedCategory = category_choice_box_menu.getSelectionModel().getSelectedItem();
            dateInstance.setCategoryName(selectedCategory);
        });
    }
}
