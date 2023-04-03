package com.example.lab_6;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.os.Handler;
import android.widget.TextView;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

public class TimeActivity extends AppCompatActivity {
   TextView mobileTime, secTime, thrTime, fourTime;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_time);
        mobileTime = findViewById(R.id.id_mobile_time);
        secTime = findViewById(R.id.id_sec_time);
        thrTime = findViewById(R.id.id_thr_time);
        fourTime = findViewById(R.id.id_fourth_time3);

        content();
    }

    public void content(){
        int count = 0;
        count++;
        setClock();
        refresh(1000);
    }

    private void refresh(int milliseconds){
        final Handler handler = new Handler();
        final Runnable runnable = new Runnable() {
            @Override
            public void run() {
                content();
            }
        };
        handler.postDelayed(runnable, milliseconds);
    }

    void setClock(){
        mobileTime.setText(""+clientDateString("GMT+1:00"));
        secTime.setText(""+clientDateString("GMT-5:00"));
        thrTime.setText(""+clientDateString("GMT"));
        fourTime.setText(""+clientDateString("GMT+9:00"));
    }

    private static String clientDateString(String country) {
        TimeZone tz = TimeZone.getTimeZone(country);
        DateFormat df = new SimpleDateFormat("HH:mm:s");
        df.setTimeZone(tz);
        return df.format(new Date());
    }
}