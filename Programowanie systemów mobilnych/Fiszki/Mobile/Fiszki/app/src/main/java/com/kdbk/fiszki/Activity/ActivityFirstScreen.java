package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import com.kdbk.fiszki.Other.InternetConnection;
import com.kdbk.fiszki.Instance.TokenInstance;
import com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI.JsonUser;
import com.kdbk.fiszki.Retrofit.Models.Login;
import com.kdbk.fiszki.Other.NextActivity;
import com.kdbk.fiszki.R;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ActivityFirstScreen extends AppCompatActivity implements View.OnClickListener {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    private boolean isBackPressedBlocked = true;
    private InternetConnection con = new InternetConnection(this);
    private NextActivity nextActivity = new NextActivity(this);
    private Button login, create, reset;
    private TextView internetError;
    private EditText loginText, passwordText;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_first_screen);
        setID();

        if(con.checkInternetConnection()) internetError.setVisibility(View.INVISIBLE);
        else internetError.setVisibility(View.VISIBLE);

        reset.setOnClickListener(this);
        create.setOnClickListener(this);
        login.setOnClickListener(this);
    }

    public void onClick(View view) {
        if(con.checkInternetConnection()){
            internetError.setVisibility(View.INVISIBLE);
            switch (view.getId()) {
                case R.id.buttonPasswordReset:
                    nextActivity.openActivity(ActivityPasswordReset.class);
                    break;
                case R.id.buttonCreate:
                    nextActivity.openActivity(ActivityRegister.class);
                    break;
                case R.id.buttonLogin:
                    checkAccount();
                    break;
            }
        }
        else internetError.setVisibility(View.VISIBLE);
    }

    private void checkAccount() {
        String loginString = String.valueOf(loginText.getText());
        String passwordString= String.valueOf(passwordText.getText());
        //String loginString = "dwin333";
        //String passwordString= "qwerty123";
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://flashcard-app-api-bkrv.onrender.com/api/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonUser jsonUser = retrofit.create(JsonUser.class);
        Login post = new Login(loginString, passwordString);
        Call<Login> call = jsonUser.login(post);
        Toast.makeText(ActivityFirstScreen.this,"Trwa logowanie", Toast.LENGTH_SHORT).show();

        call.enqueue(new Callback<Login>() {
            @Override
            public void onResponse(Call<Login> call, Response<Login> response) {
                if(response.code() == 200){
                    Login post = response.body();
                    String TokenFromRetrofit = post.getToken();
                    tokenInstance.setToken(TokenFromRetrofit);
                    tokenInstance.setUserName(loginText.getText().toString());
                    nextActivity.openActivity(ActivityMainMenu.class);
                }
                if(!response.isSuccessful()){
                    Toast.makeText(ActivityFirstScreen.this,"Błędne dane", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<Login> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityFirstScreen.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
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
        login = findViewById(R.id.buttonLogin);
        create = findViewById(R.id.buttonCreate);
        reset = findViewById(R.id.buttonPasswordReset);
        loginText = findViewById(R.id.textNick);
        passwordText = findViewById(R.id.textPassword);
        internetError = findViewById(R.id.idTextInternetErrorFirstScreen);
    }
}
