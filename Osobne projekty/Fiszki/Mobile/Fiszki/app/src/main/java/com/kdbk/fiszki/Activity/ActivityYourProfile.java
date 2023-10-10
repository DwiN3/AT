package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
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

public class ActivityYourProfile extends AppCompatActivity {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    private NextActivity nextActivity = new NextActivity(this);
    private Button yoursKitsPanel;
    private TextView textLVL, textNextLVL;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_your_profile);
        setID();
        getUserLVL();

        yoursKitsPanel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                nextActivity.openActivity(ActivityPanelKits.class);
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
                textLVL.setText(""+response.body().getLevel());
                textNextLVL.setText(response.body().getPoints()+"/"+response.body().getRequiredPoints()+" pkt");
            }

            @Override
            public void onFailure(Call<UserLVL> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityYourProfile.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
                else{
                }
            }
        });
    }

    private void setID() {
        yoursKitsPanel = findViewById(R.id.buttonYourFlashcardsProfile);
        textLVL = findViewById(R.id.textLVLProfile);
        textNextLVL = findViewById(R.id.textNextLVLProfile);
    }
}