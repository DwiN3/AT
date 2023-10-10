package app;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;
import java.io.IOException;
import java.util.Objects;

/**
 * The main application class
 */
public class App extends Application {
    private static Scene scene;

    /**
     * The function sets up the initial scene and displays the stage
     * @param stage The primary stage for the application
     * @throws IOException If an error occurs while loading the FXML file
     */
    @Override
    public void start(Stage stage) throws IOException {
        scene = new Scene(loadFXML("activity_first_screen"), 850, 580);
        stage.getIcons().add(new Image(Objects.requireNonNull(getClass().getResourceAsStream("/drawable/flashcard_icon.png"))));
        stage.setTitle("Fiszki");
        stage.setScene(scene);
        stage.show();
    }

    /**
     * Sets the root of the scene to the specified FXML file
     * @param fxml The name of the FXML file
     * @throws IOException If an error occurs while loading the FXML file
     */
    public static void setRoot(String fxml) throws IOException {scene.setRoot(loadFXML(fxml)); }

    /**
     * Loads the specified FXML file and returns the root element
     * @param fxml The name of the FXML file
     * @return The root element of the loaded FXML file
     * @throws IOException If an error occurs while loading the FXML file
     */
    public static Parent loadFXML(String fxml) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(App.class.getResource(fxml + ".fxml"));
        return fxmlLoader.load();
    }

    /**
     * The entry point of the application
     * @param args The command line arguments
     */
    public static void main(String[] args) {
        launch();
    }
}