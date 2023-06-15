package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import com.kdbk.fiszki.Other.NextActivity;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI.JsonUser;
import com.kdbk.fiszki.Retrofit.Models.Register;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ActivityRegister extends AppCompatActivity {
    private NextActivity nextActivity = new NextActivity(this);
    private Button createButton;
    private TextView emailCreateText, loginCreateText, passwordCreateText, passwordCreateReText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);
        setID();

        createButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                createAccountRetrofit();
            }
        });

    }

    private void createAccountRetrofit(){
        String loginString = String.valueOf(loginCreateText.getText());
        String emailString = String.valueOf(emailCreateText.getText());
        String passwordString= String.valueOf(passwordCreateText.getText());
        String passwordReString= String.valueOf(passwordCreateReText.getText());

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://flashcard-app-api-bkrv.onrender.com/api/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonUser jsonUser = retrofit.create(JsonUser.class);
        Register post = new Register(emailString, passwordString, passwordReString, loginString);
        Call<Register> call = jsonUser.register(post);

        call.enqueue(new Callback<Register>() {
            @Override
            public void onResponse(Call<Register> call, Response<Register> response) {
                if(response.code() == 200) Toast.makeText(ActivityRegister.this,"Udało się zarejstrować", Toast.LENGTH_SHORT).show();

                if(!response.isSuccessful()){
                    Toast.makeText(ActivityRegister.this,"Błąd w rejestracji", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<Register> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityRegister.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
                else Toast.makeText(ActivityRegister.this,"Utworzono konto pomyślnie", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void setID() {
        createButton = findViewById(R.id.buttonCreate);
        emailCreateText = findViewById(R.id.textEmailCreate);
        loginCreateText = findViewById(R.id.textLoginCreate);
        passwordCreateText = findViewById(R.id.textPasswordCreate);
        passwordCreateReText = findViewById(R.id.textPasswordCreateRe);
    }
}