package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import com.kdbk.fiszki.Instance.GameSettingsInstance;
import com.kdbk.fiszki.Other.InternetConnection;
import com.kdbk.fiszki.Other.NextActivity;
import com.kdbk.fiszki.R;

public class ActivityGameMode extends AppCompatActivity {
    private NextActivity nextActivity = new NextActivity(this);
    private InternetConnection con = new InternetConnection(this);
    private GameSettingsInstance gameSettingsInstance = GameSettingsInstance.getInstance();
    private Button quizMode, learnMode, reverse, yoursKits, categories, topWords;
    private ImageView  flagFirstImage,flagSecendImage;
    private String selectedMode = "quiz", selectedLanguage = "pl";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_game_mode);
        setID();
        setButtonListeners();

        quizMode.setBackgroundResource(R.drawable.rounded_button_mode_pressed);
        learnMode.setBackgroundResource(R.drawable.rounded_button_mode_normal);
    }

    @Override
    public void onBackPressed() {
        Intent intent = new Intent(this, ActivityMainMenu.class);
        startActivity(intent);
        finish();
    }

    private void setButtonListeners() {
            categories.setOnClickListener(v -> {
                gameSettingsInstance.setGameMode(selectedMode);
                gameSettingsInstance.setLanguage(selectedLanguage);
                nextActivity.openActivity(ActivityCategories.class);
            });
            yoursKits.setOnClickListener(v -> {
                gameSettingsInstance.setGameMode(selectedMode);
                gameSettingsInstance.setLanguage(selectedLanguage);
                nextActivity.openActivity(ActivityKits.class);
            });
            quizMode.setOnClickListener(v -> {
                selectedMode = "quiz";
                quizMode.setBackgroundResource(R.drawable.rounded_button_mode_pressed);
                learnMode.setBackgroundResource(R.drawable.rounded_button_mode_normal);
            });
            learnMode.setOnClickListener(v -> {
                selectedMode = "learn";
                learnMode.setBackgroundResource(R.drawable.rounded_button_mode_pressed);
                quizMode.setBackgroundResource(R.drawable.rounded_button_mode_normal);
            });
            reverse.setOnClickListener(v -> {
                if (selectedLanguage.equals("pl")) selectedLanguage = "ang";
                else if (selectedLanguage.equals("ang")) selectedLanguage = "pl";
                Drawable firstImage = flagFirstImage.getDrawable();
                Drawable secondImage = flagSecendImage.getDrawable();
                flagFirstImage.setImageDrawable(secondImage);
                flagSecendImage.setImageDrawable(firstImage);
            });
            topWords.setOnClickListener(v -> {
                Intent intent = new Intent(this, ActivityTopWords.class);
                startActivity(intent);
            });
        }

    private void setID() {
        quizMode = findViewById(R.id.buttonQuizMode);
        learnMode = findViewById(R.id.buttonLearnMode);
        reverse = findViewById(R.id.buttonReverse);
        flagFirstImage = findViewById(R.id.flagFirst);
        flagSecendImage = findViewById(R.id.flagSecend);
        yoursKits = findViewById(R.id.buttonYourFlashcardsMode);
        categories = findViewById(R.id.buttonCategoriesMode);
        topWords = findViewById(R.id.buttonTopWordsMode);
    }
}
