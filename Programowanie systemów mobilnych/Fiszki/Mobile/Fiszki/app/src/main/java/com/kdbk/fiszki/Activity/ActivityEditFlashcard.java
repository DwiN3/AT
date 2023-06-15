package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import com.kdbk.fiszki.Instance.FlashcardInfoInstance;
import com.kdbk.fiszki.Instance.TokenInstance;
import com.kdbk.fiszki.Other.NextActivity;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI.JsonFlashcards;
import com.kdbk.fiszki.Retrofit.Models.FlashcardID;
import java.io.IOException;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ActivityEditFlashcard extends AppCompatActivity  {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    private FlashcardInfoInstance flashcardInfoInstance = FlashcardInfoInstance.getInstance();
    private boolean isBackPressedBlocked = true;
    private NextActivity nextActivity = new NextActivity(this);
    private Button accept, back, delete;
    private EditText wordText, translateWordText,exampleText, translateExampleText;
    private String _id_word="",word, translateWord,sentens,sentensTranslate;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_edit_flashcard);
        setID();

        _id_word = flashcardInfoInstance.getId_word();
        setFlashcardRetrofit();

        accept.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                word = String.valueOf(wordText.getText());
                translateWord = String.valueOf(translateWordText.getText());
                sentens = String.valueOf(exampleText.getText());
                sentensTranslate = String.valueOf(translateExampleText.getText());
                editFlashcardRetrofit();
                nextActivity.openActivity(ActivityShowKitsEdit.class);
            }
        });
        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                nextActivity.openActivity(ActivityShowKitsEdit.class);
            }
        });

        delete.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                deleteFlashcardRetrofit();
                nextActivity.openActivity(ActivityShowKitsEdit.class);
            }
        });
    }

    public void deleteFlashcardRetrofit() {
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
        Call<FlashcardID> call = jsonFlashcards.deleteFlashcards(_id_word);

        call.enqueue(new Callback<FlashcardID>() {
            @Override
            public void onResponse(Call<FlashcardID> call, Response<FlashcardID> response) {
                if (!response.isSuccessful()) {
                    Toast.makeText(ActivityEditFlashcard.this, "Błąd operacji", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<FlashcardID> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityEditFlashcard.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
                else Toast.makeText(ActivityEditFlashcard.this, "Poprawnie usunięto fiszkę", Toast.LENGTH_SHORT).show();
            }
        });
    }

    public void editFlashcardRetrofit() {

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
        FlashcardID post = new FlashcardID(word,translateWord, sentens, sentensTranslate);
        Call<FlashcardID> call = jsonFlashcards.editFlashcards(_id_word, post);

        call.enqueue(new Callback<FlashcardID>() {
            @Override
            public void onResponse(Call<FlashcardID> call, Response<FlashcardID> response) {
                if (!response.isSuccessful()) {
                    Toast.makeText(ActivityEditFlashcard.this, "Błąd operacji", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<FlashcardID> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityEditFlashcard.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
                else Toast.makeText(ActivityEditFlashcard.this, "Poprawnie zmodyfikowano fiszkę", Toast.LENGTH_SHORT).show();
            }
        });
    }

    public void setFlashcardRetrofit() {
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
        Call<FlashcardID> call = jsonFlashcards.getFlashcard(_id_word);

        call.enqueue(new Callback<FlashcardID>() {
            @Override
            public void onResponse(Call<FlashcardID> call, Response<FlashcardID> response) {
                wordText.setText(response.body().getWord());
                translateWordText.setText(response.body().getTranslatedWord());
                exampleText.setText(response.body().getExample());
                translateExampleText.setText(response.body().getTranslatedExample());
                if (!response.isSuccessful()) {
                    Toast.makeText(ActivityEditFlashcard.this, "Błąd operacji", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<FlashcardID> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityEditFlashcard.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    public boolean dispatchKeyEvent(KeyEvent event) {
        if (event.getKeyCode() == KeyEvent.KEYCODE_BACK && isBackPressedBlocked) {
            return true;
        }
        return super.dispatchKeyEvent(event);
    }

    private void setID() {
        wordText = findViewById(R.id.word_text_edit);
        translateWordText = findViewById(R.id.translate_text_edit);
        exampleText = findViewById(R.id.example_text_edit);
        translateExampleText = findViewById(R.id.translate_example_text_edit);
        accept = findViewById(R.id.buttonEditFlashCardAccept);
        back = findViewById(R.id.buttonEditFlashCardBack);
        delete = findViewById(R.id.buttonEditFlashCardDelate);
    }
}
