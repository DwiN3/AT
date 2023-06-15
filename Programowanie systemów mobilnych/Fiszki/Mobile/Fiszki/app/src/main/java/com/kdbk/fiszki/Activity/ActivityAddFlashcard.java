package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Toast;
import com.kdbk.fiszki.Instance.TokenInstance;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI.JsonFlashcards;
import com.kdbk.fiszki.Retrofit.Models.Flashcards;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ActivityAddFlashcard extends AppCompatActivity  {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    private Button add;
    private Spinner categorySpinner;
    private EditText  kitText, wordText, translateWordText,exampleText, translateExampleText;
    private String nrKit, word, translateWord, sampleSentence, translateSampleSentence, category;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_flashcard);
        setID();
        setSpinner();

        add.setOnClickListener(new View.OnClickListener() {
           @Override
            public void onClick(View view) {
                getWord();
                addFlashcardRetrofit();
                Toast.makeText(ActivityAddFlashcard.this, "Trwa dodawanie fiszki", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void getWord(){
        nrKit = String.valueOf(kitText.getText());
        category = String.valueOf(categorySpinner.getSelectedItem());
        word = String.valueOf(wordText.getText());
        translateWord = String.valueOf(translateWordText.getText());
        sampleSentence = String.valueOf(exampleText.getText());
        translateSampleSentence = String.valueOf(translateExampleText.getText());
    }

    private void clearWord(){
        wordText.setText("");
        translateWordText.setText("");
        exampleText.setText("");
        translateExampleText.setText("");
        word = "";
        translateWord = "";
        sampleSentence = "";
        translateSampleSentence = "";
    }

    public void addFlashcardRetrofit() {
        String language = "english";

        OkHttpClient client = new OkHttpClient.Builder().addInterceptor(new Interceptor() {
            @Override
            public okhttp3.Response intercept(Chain chain) throws IOException {
                Request newRequest = chain.request().newBuilder()
                        .addHeader("Authorization", "Bearer " + tokenInstance.getToken())
                        .build();
                return chain.proceed(newRequest);
            }
        }).build();


        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://flashcard-app-api-bkrv.onrender.com/api/")
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonFlashcards jsonFlashcards = retrofit.create(JsonFlashcards.class);
        Flashcards post = new Flashcards(nrKit, language, category, word, translateWord, sampleSentence, translateSampleSentence);
        Call<Flashcards> call = jsonFlashcards.addFlashcard(nrKit, post);

        call.enqueue(new Callback<Flashcards>() {
            @Override
            public void onResponse(Call<Flashcards> call, Response<Flashcards> response) {
                if (!response.isSuccessful()) {
                    Toast.makeText(ActivityAddFlashcard.this, "Błędne dane", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<Flashcards> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityAddFlashcard.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
                else{
                    Toast.makeText(ActivityAddFlashcard.this, "Poprawnie dodano fiszkę", Toast.LENGTH_SHORT).show();
                    clearWord();
                }
            }
        });
    }

    private void setSpinner(){
        List<String> categories = new ArrayList<>();
        String[] categoriesList = {"zwierzeta","dom", "zakupy", "praca", "zdrowie", "czlowiek", "turystyka", "jedzenie","edukacja", "inne"};
        for(int n=0;n<categoriesList.length;n++){
            categories.add(categoriesList[n]);
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, categories);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        categorySpinner.setAdapter(adapter);
        categorySpinner.setSelection(adapter.getPosition("inne"));
    }

    private void setID() {
        add = findViewById(R.id.buttonAcceptFlashcard);
        kitText = findViewById(R.id.kit_text_add);
        categorySpinner = findViewById(R.id.category_spinner_add);
        wordText = findViewById(R.id.word_text_add);
        translateWordText = findViewById(R.id.translate_text_add);
        exampleText = findViewById(R.id.example_text_add);
        translateExampleText = findViewById(R.id.translate_example_text_add);
    }
}