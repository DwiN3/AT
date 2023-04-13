package com.example.lab_8;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import org.json.JSONException;
import org.json.JSONObject;
import static android.view.View.INVISIBLE;
import static android.view.View.VISIBLE;

public class MainActivity extends AppCompatActivity {
    private EditText TextCity;
    private Button buttonAccept;
    private String city, cityApp;
    private TextView TextInternetError;

    public static final String SHARED_PREFS = "sharedPrefs";
    public static final String LASTCITY = "lastcity";
    private String lastCityToSave;
    private String lastCorrectCity=" ";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setID();

        buttonAccept.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                city = TextCity.getText().toString();
                if(CheckInternetConnection()) getWeatherInfo();
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
                    cityApp = response.getString("name");
                    lastCorrectCity = city;
                    openWeatherActivity();
                } catch (JSONException e) {
                }
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(MainActivity.this, "Wrong city name", Toast.LENGTH_LONG).show();
                city = lastCorrectCity;
            }
        });
        queue.add(request);
    }

    public void openWeatherActivity(){
        if(!city.isEmpty()){
            Intent intent = new Intent(this, WeatherActivity.class);
            intent.putExtra("City", cityApp);
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

    public void updateViews(){ TextCity.setText(lastCityToSave); }

    public boolean CheckInternetConnection() {
        ConnectivityManager connectivityManager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
        if(activeNetworkInfo != null && activeNetworkInfo.isConnected()){ TextInternetError.setVisibility(INVISIBLE); return true; }
        else{ TextInternetError.setVisibility(VISIBLE); return false; }
    }

    void setID(){
        buttonAccept = findViewById(R.id.idButton);
        TextCity = findViewById(R.id.idTextCity);
        TextInternetError = findViewById(R.id.idTextInternetError);
    }
}
