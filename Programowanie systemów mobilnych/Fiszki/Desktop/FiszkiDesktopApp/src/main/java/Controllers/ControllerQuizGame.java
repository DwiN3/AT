package Controllers;

import Other.SetGame;
import Retrofit.JsonPlaceholderAPI.JsonFlashcardsCollections;
import Retrofit.Models.Flashcard;
import app.App;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.image.Image;
import Other.DateInstance;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * This class is the controller for the quiz game screen
 * Responsible for controlling the game in quiz mode
 */
public class ControllerQuizGame {
    @FXML
    private Label word_text_quiz, word_sample_text_quiz, sticks_left_quiz, userPKT_quiz;
    @FXML
    private Button answer_1_quiz, answer_2_quiz, answer_3_quiz, answer_4_quiz, next_word_button_quiz, back_menu_button_quiz;
    @FXML
    private ImageView image_quiz, image_word_quiz;
    @FXML
    private void switchActivity(String activity) throws IOException { App.setRoot(activity); }
    private Image imageWord;
    private DateInstance dateInstance = DateInstance.getInstance();
    private SetGame game;
    private int points = 0, nrWords = 0, scoreTrain = 0, bestTrain = 0, allWords = 0;
    private String selectedName = "", selectedData = "category", selectedLanguage = "pl", answer = "";
    private boolean markTheAnswer = false;
    private ArrayList<Flashcard> wordsListKit = new ArrayList<>();

    /**
     * Initializes the controller
     */
    @FXML
    public void initialize() {
        Image image = new Image(getClass().getResourceAsStream("/drawable/square_big.png"));
        image_quiz.setImage(image);

        selectedName = dateInstance.getCategoryName();
        getWordFromCateogryRetrofit();
        next_word_button_quiz.setVisible(false);

        answer_1_quiz.setOnAction(event -> {
            if (nrWords == game.getBorrder() - 1) next_word_button_quiz.setText("Podsumowanie");
            sticks_left_quiz.setText("Pozostało: " + (game.getBorrder() - nrWords-1));
            if (answer != null && answer.equals(answer_1_quiz.getText()) && !markTheAnswer) {
                answer_1_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                correctChoice();
            } else {
                inCorrectChoice();
                answer_1_quiz.setStyle("-fx-background-color: #FF0000; -fx-text-fill: white;");
                if (answer.equals(answer_2_quiz.getText())) {
                    answer_2_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                } else if (answer.equals(answer_3_quiz.getText())) {
                    answer_3_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                } else if (answer.equals(answer_4_quiz.getText())) {
                    answer_4_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                }
            }
        });

        answer_2_quiz.setOnAction(event -> {
            if (nrWords == game.getBorrder() - 1) next_word_button_quiz.setText("Podsumowanie");
            sticks_left_quiz.setText("Pozostało: " + (game.getBorrder() - nrWords-1));
            if (answer != null && answer.equals(answer_2_quiz.getText()) && !markTheAnswer) {
                answer_2_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                correctChoice();
            } else {
                inCorrectChoice();
                answer_2_quiz.setStyle("-fx-background-color: #FF0000; -fx-text-fill: white;");
                if (answer.equals(answer_1_quiz.getText())) {
                    answer_1_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                } else if (answer.equals(answer_3_quiz.getText())) {
                    answer_3_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                } else if (answer.equals(answer_4_quiz.getText()) ) {
                    answer_4_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                }
            }
        });

        answer_3_quiz.setOnAction(event -> {
            if (nrWords == game.getBorrder() - 1) next_word_button_quiz.setText("Podsumowanie");
            sticks_left_quiz.setText("Pozostało: " + (game.getBorrder() - nrWords-1));
            if (answer != null && answer.equals(answer_3_quiz.getText()) && !markTheAnswer) {
                answer_3_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                correctChoice();
            } else {
                inCorrectChoice();
                answer_3_quiz.setStyle("-fx-background-color: #FF0000; -fx-text-fill: white;");
                if (answer.equals(answer_1_quiz.getText())) {
                    answer_1_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                } else if (answer.equals(answer_2_quiz.getText())) {
                    answer_2_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                } else if (answer.equals(answer_4_quiz.getText())) {
                    answer_4_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                }
            }
        });

        answer_4_quiz.setOnAction(event -> {
            if (nrWords == game.getBorrder() - 1) next_word_button_quiz.setText("Podsumowanie");
            sticks_left_quiz.setText("Pozostało: " + (game.getBorrder() - nrWords-1));
            if (answer != null && answer.equals(answer_4_quiz.getText()) && !markTheAnswer) {
                answer_4_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                correctChoice();
            } else {
                inCorrectChoice();
                answer_4_quiz.setStyle("-fx-background-color: #FF0000; -fx-text-fill: white;");
                if (answer.equals(answer_1_quiz.getText())) {
                    answer_1_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                } else if (answer.equals(answer_2_quiz.getText())) {
                    answer_2_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                } else if (answer.equals(answer_3_quiz.getText())) {
                    answer_3_quiz.setStyle("-fx-background-color: #00FF00; -fx-text-fill: white;");
                }
            }
        });

        next_word_button_quiz.setOnAction(event -> {
            String resourcePath = "/drawable/flashcard_icon_png.png";
            Image nextImage = new Image(getClass().getResourceAsStream(resourcePath));
            image_word_quiz.setImage(nextImage);
            next_word_button_quiz.setVisible(false);
            blockButtons(false);
            if (nrWords != game.getBorrder() - 1) {
                nrWords += 1;
                clearButtons();
                setQuestion(nrWords);
            } else {
                dateInstance.setBestTrain(bestTrain);
                dateInstance.setPoints(points);
                dateInstance.setAllWords(allWords);
                try {
                    switchActivity("activity_end_screen");
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });

        back_menu_button_quiz.setOnAction(event -> {
            try {
                switchActivity("activity_main_menu");
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
        answer_1_quiz.setDisable(isLoading);
        answer_1_quiz.setOpacity(buttonOpacity);
        answer_2_quiz.setDisable(isLoading);
        answer_2_quiz.setOpacity(buttonOpacity);
        answer_3_quiz.setDisable(isLoading);
        answer_3_quiz.setOpacity(buttonOpacity);
        answer_4_quiz.setDisable(isLoading);
        answer_4_quiz.setOpacity(buttonOpacity);
    }

    /**
     * function clears button styles
     */
    public void clearButtons() {
        answer_1_quiz.setStyle("");
        answer_2_quiz.setStyle("");
        answer_3_quiz.setStyle("");
        answer_4_quiz.setStyle("");
    }

    /**
     * The function is executed after selecting the correct answer
     */
    public void correctChoice() {
        Platform.runLater(() -> {
            points += 1;
            if (scoreTrain < 0) scoreTrain = 1;
            else scoreTrain += 1;
            if (scoreTrain > bestTrain) bestTrain = scoreTrain;
            setEmoji();
            markTheAnswer = true;
            userPKT_quiz.setText("Punkty:    " + points + "/" + allWords);
            word_sample_text_quiz.setText(game.getSentenseTra(nrWords));
            next_word_button_quiz.setVisible(true);
            blockButtons(true);
        });
    }

    /**
     * The function is executed after selecting the incorrect answer
     */
    public void inCorrectChoice() {
        Platform.runLater(() -> {
            if (scoreTrain > 0) scoreTrain = -1;
            else scoreTrain--;
            setEmoji();
            markTheAnswer = true;
            userPKT_quiz.setText("Punkty:    " + points + "/" + allWords);
            word_sample_text_quiz.setText(game.getSentenseTra(nrWords));
            next_word_button_quiz.setVisible(true);
            blockButtons(true);
        });
    }

    /**
     * The function for setting emoji images
     */
    public void setEmoji() {
        String resourcePath;
        if (scoreTrain <= -5)
            resourcePath = "/drawable/emoji_m5.png";
        else if (scoreTrain == -4)
            resourcePath = "/drawable/emoji_m4.png";
        else if (scoreTrain == -3)
            resourcePath = "/drawable/emoji_m3.png";
        else if (scoreTrain == -2)
            resourcePath = "/drawable/emoji_m2.png";
        else if (scoreTrain == -1)
            resourcePath = "/drawable/emoji_m1.png";
        else if (scoreTrain == 0)
            resourcePath = "/drawable/flashcard_icon_png.png";
        else if (scoreTrain == 1)
            resourcePath = "/drawable/emoji_1.png";
        else if (scoreTrain == 2)
            resourcePath = "/drawable/emoji_2.png";
        else if (scoreTrain == 3)
            resourcePath = "/drawable/emoji_3.png";
        else if (scoreTrain == 4)
            resourcePath = "/drawable/emoji_4.png";
        else if (scoreTrain >= 5)
            resourcePath = "/drawable/emoji_5.png";
        else
            resourcePath = "/drawable/flashcard_icon_png.png";

        try (InputStream inputStream = getClass().getResourceAsStream(resourcePath)) {
            if (inputStream != null) {
                imageWord = new Image(inputStream);
                image_word_quiz.setImage(imageWord);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Question setting function
     * @param numberWord is number of word
     */
    public void setQuestion(int numberWord) {
        Platform.runLater(() -> {
            markTheAnswer = false;
            word_text_quiz.setText(game.getNameWord(numberWord));
            answer_1_quiz.setText(game.getAns1(numberWord));
            answer_2_quiz.setText(game.getAns2(numberWord));
            answer_3_quiz.setText(game.getAns3(numberWord));
            answer_4_quiz.setText(game.getAns4(numberWord));
            answer = game.getCorrectANS(numberWord);
            word_sample_text_quiz.setText(game.getSentense(numberWord));
        });
    }

    /**
     * Retrieve a set of words using Retrofit
     */
    public void getWordFromCateogryRetrofit() {
        wordsListKit.clear();
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

        JsonFlashcardsCollections jsonFlashcardsCollections = retrofit.create(JsonFlashcardsCollections.class);
        Call<List<List<Flashcard>>> call = jsonFlashcardsCollections.getCategory(selectedName);

        call.enqueue(new Callback<List<List<Flashcard>>>() {
            @Override
            public void onResponse(Call<List<List<Flashcard>>> call, Response<List<List<Flashcard>>> response) {
                if (response.isSuccessful()) {
                    List<List<Flashcard>> elementLists = response.body();
                    if (elementLists != null) {
                        processElements(elementLists);
                        game = new SetGame(selectedData,"quiz", selectedLanguage, wordsListKit);
                        scoreTrain = 0;
                        nrWords = 0;
                        allWords = game.getListSize();
                        Platform.runLater(() -> {
                            userPKT_quiz.setText("Punkty:    " + points + "/" + allWords);
                            sticks_left_quiz.setText("Pozostało: "+game.getBorrder());
                        });
                        setEmoji();
                        setQuestion(nrWords);
                    }
                }
            }

            @Override
            public void onFailure(Call<List<List<Flashcard>>> call, Throwable t) {}
        });
    }

    /**
     * The fetched words function adds to the list of all words
     * @param elementLists is list of item lists
     */
    public void processElements(List<List<Flashcard>> elementLists) {
        for (List<Flashcard> elementList : elementLists) {
            int id_count=0;
            for (Flashcard element : elementList) {
                wordsListKit.add(new Flashcard(element.getWord(), element.getTranslatedWord(), element.getExample(), element.getTranslatedExample(), id_count, element.get_id()));
                id_count++;
            }
        }
    }
}
