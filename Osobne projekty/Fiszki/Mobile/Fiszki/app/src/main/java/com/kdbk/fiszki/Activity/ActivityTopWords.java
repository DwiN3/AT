package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import com.kdbk.fiszki.Instance.GameSettingsInstance;
import com.kdbk.fiszki.Instance.TokenInstance;
import com.kdbk.fiszki.Other.NextActivity;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.RecyclerView.Adaper.AdapterTopWords;
import com.kdbk.fiszki.RecyclerView.Model.ModelKits;
import com.kdbk.fiszki.RecyclerView.Model.ModelShowKitsEdit;
import com.kdbk.fiszki.RecyclerView.SelectListener.SelectListenerTopWords;
import com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI.JsonFlashcardsCollections;
import com.kdbk.fiszki.Retrofit.Models.FlashcardCollections;
import com.kdbk.fiszki.Retrofit.Models.FlashcardCollectionsWords;
import com.kdbk.fiszki.Retrofit.Models.FlashcardID;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ActivityTopWords extends AppCompatActivity implements SelectListenerTopWords {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    private GameSettingsInstance gameSettingsInstance = GameSettingsInstance.getInstance();
    private NextActivity nextActivity = new NextActivity(this);
    private RecyclerView mRecyclerView;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager mLayoutManager;
    private Button back;
    private ArrayList<ModelShowKitsEdit> wordsList = new ArrayList<>();
    private ArrayList<ModelKits> collectionList = new ArrayList<>();
    private String listToShow="";
    private boolean isBackPressedBlocked = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_top_words);
        setID();
        getKitName();
        clearGameSettings();

        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                nextActivity.openActivity(ActivityGameMode.class);
            }
        });
    }

    private void getKitName() {
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
            public void onResponse(Call<List<FlashcardCollections>> call, Response<List<FlashcardCollections>> response) {
                List<FlashcardCollections> list = response.body();
                if (list == null || list.isEmpty()) {
                    return;
                }
                else{
                    int id = 0;
                    for (FlashcardCollections collection : list) {
                        collectionList.add(new ModelKits(collection.getCollectionName(), "ILOSC FISZEK", collection.getFlashcardsSize(), id, 30, collection.getId()));
                        id++;
                    }
                    Random random = new Random();
                    int randomNumber = random.nextInt(list.size());
                    listToShow = collectionList.get(randomNumber).getNameKit();
                    showWords();
                }
            }

            @Override
            public void onFailure(Call<List<FlashcardCollections>> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityTopWords.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
            }
        });
    }


    public void showWords() {
        wordsList.clear();
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
        Call<FlashcardCollectionsWords> call = jsonFlashcardsCollections.getKit(listToShow);

        call.enqueue(new Callback<FlashcardCollectionsWords>() {
            @Override
            public void onResponse(Call<FlashcardCollectionsWords> call, Response<FlashcardCollectionsWords> response) {
                if (response.isSuccessful()) {
                    FlashcardCollectionsWords flashcardCollection = response.body();

                    if (flashcardCollection != null) {
                        ArrayList<FlashcardID> kitWordsList = flashcardCollection.getFlashcards();
                        if (kitWordsList != null && !kitWordsList.isEmpty()) {
                            int id_count=0;
                            for (FlashcardID collection : kitWordsList) {
                                //System.out.println(collection.getWord() + "    " + collection.getTranslatedWord()+ "    " + collection.getExample() + "    " +collection.getTranslatedExample() + "    " +id_count+ "   "+ collection.get_id());
                                wordsList.add(new ModelShowKitsEdit(collection.getWord(), collection.getTranslatedWord(), collection.getExample(), collection.getTranslatedExample(),id_count, collection.get_id()));
                                id_count++;
                            }
                            RefreshRecycleView();
                        }
                    }
                } else {
                    Toast.makeText(ActivityTopWords.this, "Błąd danych", Toast.LENGTH_SHORT).show();
                }
                RefreshRecycleView();
            }

            @Override
            public void onFailure(Call<FlashcardCollectionsWords> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityTopWords.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
            }
        });
    }


    private void RefreshRecycleView() {
        mRecyclerView = findViewById(R.id.showTopWordRecycleView);
        mRecyclerView.setHasFixedSize(true);
        mLayoutManager = new LinearLayoutManager(this);
        mAdapter = new AdapterTopWords(wordsList, this);
        mRecyclerView.setLayoutManager(mLayoutManager);
        mRecyclerView.setAdapter(mAdapter);
    }

    private void clearGameSettings(){
        gameSettingsInstance.setLanguage("");
        gameSettingsInstance.setGameMode("");
        gameSettingsInstance.setName("");
        gameSettingsInstance.setSelectData("");
        gameSettingsInstance.setName("");
        gameSettingsInstance.setBestTrain(0);
        gameSettingsInstance.setPoints(0);
        gameSettingsInstance.setAllWords(0);
    }


    @Override
    public boolean dispatchKeyEvent(KeyEvent event) {
        if (event.getKeyCode() == KeyEvent.KEYCODE_BACK && isBackPressedBlocked) {
            return true;
        }
        return super.dispatchKeyEvent(event);
    }
    @Override
    public void onItemClicked(ModelShowKitsEdit modelShowKitsEdit) {

    }
    private void setID() {
        back = findViewById(R.id.buttonTopWordBack);
    }
}