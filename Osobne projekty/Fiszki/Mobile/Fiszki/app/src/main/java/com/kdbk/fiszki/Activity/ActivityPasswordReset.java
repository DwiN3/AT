package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
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

public class ActivityPasswordReset extends AppCompatActivity {

    private NextActivity nextActivity = new NextActivity(this);
    private EditText email, passwordConfirm, passwordConfirmRe;
    private Button resetPassword;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_password_reset);
        setID();

        resetPassword.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                resetPasswordRetrofit();
            }
        });
    }

    private void resetPasswordRetrofit(){
        String emailString = String.valueOf(email.getText());
        String passwordString= String.valueOf(passwordConfirm.getText());
        String passwordReString= String.valueOf(passwordConfirmRe.getText());

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://flashcard-app-api-bkrv.onrender.com/api/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonUser jsonUser = retrofit.create(JsonUser.class);
        Register post = new Register(emailString,passwordString, passwordReString);
        Call<String> call = jsonUser.resetPassword(post);

        call.enqueue(new Callback<String>() {
            @Override
            public void onResponse(Call<String> call, Response<String> response) {
                if(response.code() == 200) Toast.makeText(ActivityPasswordReset.this,"Udało się zmienić hasło", Toast.LENGTH_SHORT).show();

                if(!response.isSuccessful()){
                    Toast.makeText(ActivityPasswordReset.this,"Błąd w resecie hasła", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<String> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityPasswordReset.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void setID() {
        email = findViewById(R.id.textEmail);
        passwordConfirm = findViewById(R.id.textPasswordConfirm);
        passwordConfirmRe = findViewById(R.id.textPasswordConfirmRe);
        resetPassword = findViewById(R.id.buttonPasswordResetConfirm);
    }
}