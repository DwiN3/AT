package com.example.lab_9;

import static android.view.View.INVISIBLE;
import static android.view.View.VISIBLE;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class SettingsActivity extends AppCompatActivity {
    private EditText TextName;
    private Button buttonAccept;
    private String login;
    private RadioGroup buttons;
    private Button showType1;
    private boolean choiceToLimitMessage = false;
    private TextView TextInternetError;

    public static final String SHARED_PREFS = "sharedPrefs";
    public static final String LASTLOGIN = "lastlogin";
    private String lastLoginToSave;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
        setID();

        if(!checkInternetConnection()) TextInternetError.setVisibility(VISIBLE);

        buttonAccept.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                login = TextName.getText().toString();
                if(checkInternetConnection()){
                    if(login.isEmpty()) Toast.makeText(SettingsActivity.this,"The username is invalid", Toast.LENGTH_SHORT).show();
                    else {
                        openMainMenu();
                        saveData();
                    }
                }
            }
        });
        loadData();
        updateViews();
    }

    public void openMainMenu() {
        setNumberOfMessages();
        Intent intent = new Intent(this, MainMenuActivity.class);
        intent.putExtra("Choice", choiceToLimitMessage);
        intent.putExtra("Login", login);
        startActivity(intent);
    }

    public void setNumberOfMessages(){
        int radioID = buttons.getCheckedRadioButtonId();
        if(radioID == showType1.getId()) choiceToLimitMessage = true;
        else choiceToLimitMessage = false;
    }

    public void updateViews() {
        TextName.setText(lastLoginToSave);
    }

    public void saveData() {
        SharedPreferences sharedPreferences = getSharedPreferences(SHARED_PREFS, MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString(LASTLOGIN, TextName.getText().toString());
        editor.apply();
    }

    public void loadData() {
        SharedPreferences sharedPreferences = getSharedPreferences(SHARED_PREFS, MODE_PRIVATE);
        lastLoginToSave = sharedPreferences.getString(LASTLOGIN, "");
    }

    public boolean checkInternetConnection() {
        ConnectivityManager connectivityManager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
        if(activeNetworkInfo != null && activeNetworkInfo.isConnected()){ TextInternetError.setVisibility(INVISIBLE); return true; }
        else{ TextInternetError.setVisibility(VISIBLE); return false; }
    }

    public void setID() {
        buttonAccept = findViewById(R.id.id_buttonAccept);
        TextName = findViewById(R.id.id_textCity);
        buttons = findViewById(R.id.id_radioGroup);
        showType1 = findViewById(R.id.id_radioOne);
        TextInternetError = findViewById(R.id.idTextInternetError);
    }
}
