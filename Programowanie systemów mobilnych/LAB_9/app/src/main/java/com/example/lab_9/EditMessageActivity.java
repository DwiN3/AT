package com.example.lab_9;

import static android.view.View.INVISIBLE;
import static android.view.View.VISIBLE;

import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class EditMessageActivity extends AppCompatActivity {
    private String login, date, content, actualLogin, id;
    private String loginEdited, contentEditied;
    private TextView TextLogin, TextDate, TextEditCheck;
    private EditText messageEdit;
    private Button acceptButton, deleteButton;
    private boolean choiceToLimitMessage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_edit_message);
        setID();
        setValues();
        setScreen();

        acceptButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(!checkInternetConnection()) openSettings();
                else{
                    loginEdited = TextLogin.getText().toString();
                    contentEditied = messageEdit.getText().toString();
                    if(checkLogin()) editMessage();
                    Intent intent = new Intent(EditMessageActivity.this, MainMenuActivity.class);
                    intent.putExtra("Login", actualLogin);
                    startActivity(intent);
                }
            }
        });
        deleteButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(!checkInternetConnection()) openSettings();
                else{
                    if(checkLogin()) deleteMessage();
                    Intent intent = new Intent(EditMessageActivity.this, MainMenuActivity.class);
                    intent.putExtra("Login", actualLogin);
                    intent.putExtra("Choice", choiceToLimitMessage);
                    startActivity(intent);
                }
            }
        });
    }

    public void setValues(){
        Intent intent = getIntent();
        login = intent.getStringExtra("EditLogin");
        date = intent.getStringExtra("EditDate");
        content = intent.getStringExtra("EditContent");
        actualLogin = intent.getStringExtra("ActualLogin");
        id = intent.getStringExtra("EditID");
        choiceToLimitMessage = intent.getExtras().getBoolean("Choice");
        loginEdited =  login;
        contentEditied = content;
    }

    public boolean checkLogin(){ return actualLogin.equals(login); }

    public void setScreen(){
        if(checkLogin()){
            TextEditCheck.setText("You can edit this message");
            TextEditCheck.setTextColor(Color.GREEN);
            acceptButton.setText("EDIT");
            deleteButton.setVisibility(VISIBLE);
        }
        else{
            TextEditCheck.setText("You do not have permission to edit this message");
            TextEditCheck.setTextColor(Color.RED);
            acceptButton.setText("RETURN");
            deleteButton.setVisibility(INVISIBLE);
        }
        TextLogin.setText(login);
        TextLogin.setBackground(null);
        TextDate.setText(date);
        messageEdit.setText(content);
    }

    public void editMessage(){
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://tgryl.pl/shoutbox/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonPlaceholderAPI jsonPlaceholderAPI = retrofit.create(JsonPlaceholderAPI.class);
        Post post = new Post(loginEdited, contentEditied);
        Call<List<Post>> call = jsonPlaceholderAPI.putPost(id,post);

        call.enqueue(new Callback<List<Post>>() {
            @Override
            public void onResponse(Call<List<Post>> call, Response<List<Post>> response) {
                Toast.makeText(EditMessageActivity.this,"Code: "+response.code(), Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onFailure(Call<List<Post>> call, Throwable t) {}
        });
    }

    public void deleteMessage(){
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://tgryl.pl/shoutbox/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonPlaceholderAPI jsonPlaceholderAPI = retrofit.create(JsonPlaceholderAPI.class);
        Call<List<Post>> call = jsonPlaceholderAPI.deletePost(id);

        call.enqueue(new Callback<List<Post>>() {
            @Override
            public void onResponse(Call<List<Post>> call, Response<List<Post>> response) {
                Toast.makeText(EditMessageActivity.this,"Code: "+response.code(), Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onFailure(Call<List<Post>> call, Throwable t) {}
        });
    }

    public boolean checkInternetConnection() {
        ConnectivityManager connectivityManager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
        if(activeNetworkInfo != null && activeNetworkInfo.isConnected()) return true;
        else return false;
    }

    public void openSettings(){
        Intent intent = new Intent(this, SettingsActivity.class);
        startActivity(intent);
    }

    public void setID() {
        acceptButton = findViewById(R.id.id_messageEditButton);
        TextLogin = findViewById(R.id.id_messageEditLogin);
        TextDate = findViewById(R.id.id_messageEditDate);
        messageEdit = findViewById(R.id.id_messageEditContent);
        TextEditCheck = findViewById(R.id.id_editCheck);
        deleteButton = findViewById(R.id.id_deleteButton);
    }
}