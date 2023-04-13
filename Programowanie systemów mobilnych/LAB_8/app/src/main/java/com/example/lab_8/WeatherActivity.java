package com.example.lab_8;

import androidx.appcompat.app.AppCompatActivity;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;
import android.content.Context;
import android.content.Intent;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.os.Handler;
import android.widget.ImageView;
import android.widget.TextView;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.squareup.picasso.Picasso;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;
import java.util.TimeZone;

public class WeatherActivity extends AppCompatActivity {
    TextView TextCity, TextTime, TextTemp, TextPressure, TextHumidity, TextTempMin, TextTempMax, TextRandomNumber;
    String city, temp, pressure, humidity, tempMin, tempMax, timeZone, randomNumber, iconNumber;
    int timeZoneForCity;
    SwipeRefreshLayout swipeRefreshLayout;
    ImageView iconWeather;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_weather);
        setID();

        Intent intent = getIntent();
        city = intent.getStringExtra("City");

        getWeatherInfo();

        swipeRefreshLayout.setOnRefreshListener(new SwipeRefreshLayout.OnRefreshListener() {
            @Override
            public void onRefresh() {
                if(CheckInternetConnection()) getWeatherInfo();
                swipeRefreshLayout.setRefreshing(false);
            }
        });
        contentRefresh5Min();
    }

    public void contentRefresh5Min(){
        int count = 0;
        count++;
        if(CheckInternetConnection()) getWeatherInfo();
        refresh(300000);
    }
    private void refresh(int milliseconds){
        final Handler handler = new Handler();
        final Runnable runnable = new Runnable() {
            @Override
            public void run() { contentRefresh5Min(); }
        };
        handler.postDelayed(runnable, milliseconds);
    }

    private static String clientDateString(String country) {
        TimeZone tz = TimeZone.getTimeZone(country);
        DateFormat df = new SimpleDateFormat("HH:mm");
        df.setTimeZone(tz);
        return df.format(new Date());
    }

    public void getWeatherInfo(){
        RequestQueue queue = Volley.newRequestQueue(getApplicationContext());
        String url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&units=metric&APPID=749561a315b14523a8f5f1ef95e45864";
        JsonObjectRequest request = new JsonObjectRequest(Request.Method.GET, url,null, new Response.Listener<JSONObject>() {

            @Override
            public void onResponse(JSONObject response) {
                try {
                    JSONObject arrayMain= (JSONObject) response.get("main");
                    temp = arrayMain.getString("temp");
                    pressure = arrayMain.getString("pressure");
                    humidity = arrayMain.getString("humidity");
                    tempMin = arrayMain.getString("temp_min");
                    tempMax = arrayMain.getString("temp_max");
                    timeZone = response.getString("timezone");
                    JSONArray arrayIcon = (JSONArray) response.get("weather");
                    for (int i = 0; i < arrayIcon.length(); i++) {
                        iconNumber = arrayIcon.getJSONObject(i).getString("icon");
                    }
                    timeZoneForCity = (Integer.valueOf(timeZone))/3600;
                    Random rand = new Random();
                    randomNumber = String.valueOf(rand.nextInt(10+1));
                    setScreens();
                } catch (JSONException e) {}
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
            }
        });
        queue.add(request);
    }

    void setScreens(){
        TextCity.setText(city);
        TextTemp.setText(temp+"℃");
        TextPressure.setText(pressure+" hpa");
        TextHumidity.setText(humidity+" %");
        TextTempMin.setText(tempMin+"℃");
        TextTempMax.setText(tempMax+"℃");
        TextRandomNumber.setText(""+randomNumber);
        TextTime.setText(""+clientDateString("GMT+"+timeZoneForCity));
        setIconWeather();
    }

    void setIconWeather(){
        String url = "https://openweathermap.org/img/wn/"+iconNumber+"@2x.png";
        Picasso.with(this).load(url).into(iconWeather);
    }

    public boolean CheckInternetConnection() {
        ConnectivityManager connectivityManager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
        if(activeNetworkInfo != null && activeNetworkInfo.isConnected()) return true;
        else return false;
    }

    void setID(){
        TextCity = findViewById(R.id.id_city);
        TextTime = findViewById(R.id.id_Time);
        TextTemp = findViewById(R.id.id_Temp);
        TextPressure = findViewById(R.id.id_Pressure);
        TextHumidity = findViewById(R.id.id_Humidity);
        TextTempMin = findViewById(R.id.id_TempMin);
        TextTempMax = findViewById(R.id.id_TempMax);
        TextRandomNumber = findViewById(R.id.idRandomNumber);
        swipeRefreshLayout = findViewById(R.id.id_swipeRefreshLayout);
        iconWeather = findViewById(R.id.id_icon);
    }
}

