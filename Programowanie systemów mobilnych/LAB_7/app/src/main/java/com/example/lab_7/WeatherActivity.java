package com.example.lab_7;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.widget.TextView;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

public class WeatherActivity extends AppCompatActivity {
    TextView TextCity, TextTime, TextTemp, TextPressure, TextHumidity, TextTempMin, TextTempMax;
    private String nameApp, tempApp, pressureApp, humidityApp, tempMinApp, tempMaxApp, timeZoneApp;
    int timeZoneForCity;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_weather);
        setID();

        Intent intent = getIntent();
        nameApp = intent.getStringExtra("City");
        tempApp = intent.getStringExtra("Temp");
        pressureApp = intent.getStringExtra("Pressure");
        humidityApp = intent.getStringExtra("Humidity");
        tempMinApp = intent.getStringExtra("Temp_min");
        tempMaxApp = intent.getStringExtra("Temp_max");
        timeZoneApp = intent.getStringExtra("TimeZone");

        Integer timeZoneValue = Integer.valueOf(timeZoneApp);
        timeZoneForCity = (timeZoneValue/3600);

        setScreens();
        contentRefresh();
    }

    void setScreens(){
        TextCity.setText(nameApp);
        TextTemp.setText(tempApp+"℃");
        TextPressure.setText(pressureApp+" hpa");
        TextHumidity.setText(humidityApp+" %");
        TextTempMin.setText(tempMinApp+"℃");
        TextTempMax.setText(tempMaxApp+"℃");
    }

    public void contentRefresh(){
        int count = 0;
        count++;
        setClock();
        refresh(30000);
    }

    private void refresh(int milliseconds){
        final Handler handler = new Handler();
        final Runnable runnable = new Runnable() {
            @Override
            public void run() {
                contentRefresh();
            }
        };
        handler.postDelayed(runnable, milliseconds);
    }

    void setClock(){
        TextTime.setText(""+clientDateString("GMT+"+timeZoneForCity));
    }

    private static String clientDateString(String country) {
        TimeZone tz = TimeZone.getTimeZone(country);
        DateFormat df = new SimpleDateFormat("HH:mm");
        df.setTimeZone(tz);
        return df.format(new Date());
    }

    void setID(){
        TextCity = findViewById(R.id.id_city);
        TextTime = findViewById(R.id.id_Time);
        TextTemp = findViewById(R.id.id_Temp);
        TextPressure = findViewById(R.id.id_Pressure);
        TextHumidity = findViewById(R.id.id_Humidity);
        TextTempMin = findViewById(R.id.id_TempMin);
        TextTempMax = findViewById(R.id.id_TempMax);
    }
}

