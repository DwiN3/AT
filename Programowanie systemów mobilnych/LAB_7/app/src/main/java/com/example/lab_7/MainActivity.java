package com.example.lab_7;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import org.json.JSONException;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity {
    private EditText TextCity;
    private Button buttonAccept;
    private String city;
    private String nameApp, tempApp, pressureApp, humidityApp, tempMinApp, tempMaxApp, timeZoneApp;

    public static final String SHARED_PREFS = "sharedPrefs";
    public static final String LASTCITY = "lastcity";
    private String lastCityToSave;
    private String lastCorrectCity=" ";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        buttonAccept = findViewById(R.id.idButton);
        TextCity = findViewById(R.id.idTextCity);

        buttonAccept.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                city = TextCity.getText().toString();
                getWeatherInfo();
                saveData();
            }
        });
        loadData();
        updateViews();
    }
    public void getWeatherInfo(){
        RequestQueue queue = Volley.newRequestQueue(getApplicationContext());
        String url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&units=metric&APPID=749561a315b14523a8f5f1ef95e45864";
        JsonObjectRequest request = new JsonObjectRequest(Request.Method.GET, url,null, new Response.Listener<JSONObject>() {

            @Override
            public void onResponse(JSONObject response) {
                try {
                    JSONObject arrayMain= (JSONObject) response.get("main");
                    tempApp = arrayMain.getString("temp");
                    pressureApp = arrayMain.getString("pressure");
                    humidityApp = arrayMain.getString("humidity");
                    tempMinApp = arrayMain.getString("temp_min");
                    tempMaxApp = arrayMain.getString("temp_max");
                    timeZoneApp = response.getString("timezone");
                    nameApp = response.getString("name");
                    lastCorrectCity = city;
                    openWeatherActivity();
                } catch (JSONException e) {
                }
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                city = lastCorrectCity;
                getWeatherInfo();
            }
        });
        queue.add(request);
    }

    public void openWeatherActivity(){
        if(!city.isEmpty()){
            Intent intent = new Intent(this, WeatherActivity.class);
            intent.putExtra("City", nameApp);
            intent.putExtra("Temp", tempApp);
            intent.putExtra("Pressure", pressureApp);
            intent.putExtra("Humidity", humidityApp);
            intent.putExtra("Temp_min", tempMinApp);
            intent.putExtra("Temp_max", tempMaxApp);
            intent.putExtra("TimeZone", timeZoneApp);
            startActivity(intent);
        }
    }

    public void saveData(){
        SharedPreferences sharedPreferences = getSharedPreferences(SHARED_PREFS, MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString(LASTCITY, TextCity.getText().toString());
        editor.apply();
    }
    public void loadData(){
        SharedPreferences sharedPreferences = getSharedPreferences(SHARED_PREFS, MODE_PRIVATE);
        lastCityToSave = sharedPreferences.getString(LASTCITY,"");
    }

    public void updateViews(){
        TextCity.setText(lastCityToSave);
    }
}