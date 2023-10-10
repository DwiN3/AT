package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import com.kdbk.fiszki.Instance.GameSettingsInstance;
import com.kdbk.fiszki.Instance.TokenInstance;
import com.kdbk.fiszki.Other.NextActivity;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI.JsonUser;
import com.kdbk.fiszki.Retrofit.Models.UserLVL;
import java.io.IOException;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ActivityQuizEnd extends AppCompatActivity {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    private GameSettingsInstance gameSettingsInstance = GameSettingsInstance.getInstance();
    private NextActivity nextActivity = new NextActivity(this);
    private Button exit;
    private TextView ScoreEndQuiz,userBestTrainQuiz, LVLEndQuiz, NextLVLEndQuiz;
    private int bestTrain=0, points =0, allWords=0;
    private boolean isBackPressedBlocked = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_quiz_end);
        setID();
        if(gameSettingsInstance.getSelectData().equals("category")){
            gameSettingsInstance.setName("inne");
        }

        points = (gameSettingsInstance.getPoints()*10);
        sendPoints();

        allWords = gameSettingsInstance.getAllWords();
        bestTrain = gameSettingsInstance.getBestTrain();
        ScoreEndQuiz.setText(points+"/"+String.valueOf(allWords*10)+" pkt");
        userBestTrainQuiz.setText(String.valueOf(bestTrain));

        exit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                nextActivity.openActivity(ActivityMainMenu.class);
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

    private void sendPoints() {
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
        JsonUser jsonUser = retrofit.create(JsonUser.class);
        UserLVL pkt = new UserLVL(points);
        Call<UserLVL> call = jsonUser.points(pkt);

        call.enqueue(new Callback<UserLVL>() {
            @Override
            public void onResponse(Call<UserLVL> call, Response<UserLVL> response) {
                if(response.isSuccessful()){
                    //System.out.println("Wysłano "+points);
                }
            }

            @Override
            public void onFailure(Call<UserLVL> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityQuizEnd.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
                else{
                    getUserLVL();
                }
            }
        });
    }

    private void getUserLVL() {
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
        JsonUser jsonUser = retrofit.create(JsonUser.class);
        Call<UserLVL> call = jsonUser.getUserLVL();


        call.enqueue(new Callback<UserLVL>() {
            @Override
            public void onResponse(Call<UserLVL> call, Response<UserLVL> response) {
                LVLEndQuiz.setText(""+response.body().getLevel());
                NextLVLEndQuiz.setText(response.body().getPoints()+"/"+response.body().getRequiredPoints()+" pkt");
            }

            @Override
            public void onFailure(Call<UserLVL> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityQuizEnd.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
                else{
                }
            }
        });
    }



    private void setID() {
        exit = findViewById(R.id.buttonBackToMenuEndQuiz);
        ScoreEndQuiz = findViewById(R.id.textScoreEndQuiz);
        userBestTrainQuiz = findViewById(R.id.userBestTrainQuizEndText);
        LVLEndQuiz = findViewById(R.id.textLVLEndQuiz);
        NextLVLEndQuiz = findViewById(R.id.textNextLVLEndQuiz);
    }
}