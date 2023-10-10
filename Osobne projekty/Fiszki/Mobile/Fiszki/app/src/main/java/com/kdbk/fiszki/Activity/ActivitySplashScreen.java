package com.kdbk.fiszki.Activity;

import static com.kdbk.fiszki.Activity.ActivityMainMenu.LASTTOKEN;
import static com.kdbk.fiszki.Activity.ActivityMainMenu.LASTUSERNAME;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import com.kdbk.fiszki.Instance.TokenInstance;
import com.kdbk.fiszki.R;

public class ActivitySplashScreen extends AppCompatActivity {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    public static final String SHARED_PREFS = "sharedPrefs";
    private Class<?> startScreen;
    private String TOKEN = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash_screen);
        loadData();

        if (!tokenInstance.getToken().equals(TOKEN)) {
            startScreen = ActivityMainMenu.class;
        } else {
            startScreen = ActivityFirstScreen.class;
        }

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                if (startScreen != null) {
                    Intent intent = new Intent(ActivitySplashScreen.this, startScreen);
                    startActivity(intent);
                } else {
                    // Handle error
                }
                finish();
            }
        }, 2500);
    }

    public void loadData() {
        SharedPreferences sharedPreferences = getSharedPreferences(SHARED_PREFS, MODE_PRIVATE);
        String username = sharedPreferences.getString(LASTUSERNAME, "");
        String ttoken = sharedPreferences.getString(LASTTOKEN, "");
        tokenInstance.setUserName(username);
        tokenInstance.setToken(ttoken);
    }
}
