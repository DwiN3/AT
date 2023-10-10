package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;
import com.kdbk.fiszki.Instance.GameSettingsInstance;
import com.kdbk.fiszki.Instance.TokenInstance;
import com.kdbk.fiszki.RecyclerView.Adaper.AdapterKits;
import com.kdbk.fiszki.RecyclerView.Model.ModelKits;
import com.kdbk.fiszki.Other.NextActivity;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.RecyclerView.SelectListener.SelectListenerKits;
import com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI.JsonFlashcardsCollections;
import com.kdbk.fiszki.Retrofit.Models.FlashcardCollections;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ActivityKits extends AppCompatActivity implements SelectListenerKits {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    private GameSettingsInstance gameSettingsInstance = GameSettingsInstance.getInstance();
    private NextActivity nextActivity = new NextActivity(this);
    private RecyclerView mRecyclerView;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager mLayoutManager;
    private TextView noKitsInfo;
    private ArrayList<ModelKits> collectionList = new ArrayList<>();
    private String selectedMode = "";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_your_kits);
        setID();

        selectedMode = gameSettingsInstance.getGameMode();

        fetchFlashcardsCategoriesRetrofit();
    }

    @Override
    public void onItemClicked(ModelKits modelKits) {
        gameSettingsInstance.setName(modelKits.getNameKit());
        gameSettingsInstance.setSelectData("kit");


        if(selectedMode.equals("quiz") && modelKits.getNumberOfCards()>= gameSettingsInstance.getBorderMinFlashcardQuiz()){
            nextActivity.openActivity(ActivityQuizScreen.class);
        } else if(selectedMode.equals("learn")){
            nextActivity.openActivity(ActivityLearningScreen.class);
        }

        else Toast.makeText(ActivityKits.this, "Zestaw zawiera za mało fiszek", Toast.LENGTH_SHORT).show();
    }

    private void fetchFlashcardsCategoriesRetrofit() {
        collectionList.clear();
        OkHttpClient client = new OkHttpClient.Builder().addInterceptor(new Interceptor() {
            @Override
            public okhttp3.Response intercept(Chain chain) throws IOException {
                Request newRequest = chain.request().newBuilder()
                        .addHeader("Authorization", "Bearer " + tokenInstance.getToken())
                        .build();
                return chain.proceed(newRequest);
            }
        }).build();


        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://flashcard-app-api-bkrv.onrender.com/api/")
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonFlashcardsCollections jsonFlashcardsCollections = retrofit.create(JsonFlashcardsCollections.class);
        Call<List<FlashcardCollections>> call = jsonFlashcardsCollections.getAllFlashcardsCollections();

        call.enqueue(new Callback<List<FlashcardCollections>>() {
            @Override
            public void onResponse(Call <List<FlashcardCollections>> call, Response<List<FlashcardCollections>> response) {
                List<FlashcardCollections> list = response.body();
                int countID = 0;
                for(FlashcardCollections collection : list){
                    collectionList.add(new ModelKits(collection.getCollectionName(), "ILOSC FISZEK", collection.getFlashcardsSize(), countID, 30,collection.getId()));
                    countID++;
                }
                RefreshRecycleView();
                if (!response.isSuccessful()) {
                    Toast.makeText(ActivityKits.this, "Błędne dane", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<List<FlashcardCollections>> call, Throwable t) {
                if(t.getMessage().equals("timeout")){
                    noKitsInfo.setText("BRAK DANYCH");
                    noKitsInfo.setVisibility(View.VISIBLE);
                    Toast.makeText(ActivityKits.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
    private void RefreshRecycleView(){
        mRecyclerView = findViewById(R.id.kitsRecycleView);
        mRecyclerView.setHasFixedSize(true);
        mLayoutManager = new LinearLayoutManager(this);
        mAdapter = new AdapterKits(collectionList, this, R.layout.recycler_view_kits);
        mRecyclerView.setLayoutManager(mLayoutManager);
        mRecyclerView.setAdapter(mAdapter);
        if(collectionList.isEmpty()) noKitsInfo.setVisibility(View.VISIBLE);
        else noKitsInfo.setVisibility(View.GONE);
    }



    private void setID() {
        noKitsInfo = findViewById(R.id.textNoKits);
    }
}