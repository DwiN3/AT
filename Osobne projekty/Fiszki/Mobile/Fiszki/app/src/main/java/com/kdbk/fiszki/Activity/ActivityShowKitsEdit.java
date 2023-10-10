package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import com.kdbk.fiszki.Instance.FlashcardInfoInstance;
import com.kdbk.fiszki.Instance.TokenInstance;
import com.kdbk.fiszki.RecyclerView.Adaper.AdapterShowKitsEdit;
import com.kdbk.fiszki.RecyclerView.Model.ModelShowKitsEdit;
import com.kdbk.fiszki.Other.NextActivity;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.RecyclerView.SelectListener.SelectListenerShowKitsEdit;
import com.kdbk.fiszki.Retrofit.JsonPlaceholderAPI.JsonFlashcardsCollections;
import com.kdbk.fiszki.Retrofit.Models.FlashcardCollectionsWords;
import com.kdbk.fiszki.Retrofit.Models.FlashcardID;
import java.io.IOException;
import java.util.ArrayList;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ActivityShowKitsEdit extends AppCompatActivity implements SelectListenerShowKitsEdit {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    private FlashcardInfoInstance flashcardInfoInstance = FlashcardInfoInstance.getInstance();
    private NextActivity nextActivity = new NextActivity(this);
    private RecyclerView mRecyclerView;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager mLayoutManager;
    private Button back;
    private ArrayList<ModelShowKitsEdit> wordsList = new ArrayList<>();
    private String selectedMode = "", selectedLanguage = "", nameKit="";
    private boolean isBackPressedBlocked = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_show_kits_edit);
        setID();

        nameKit = flashcardInfoInstance.getNameCollection();
        showKits();

        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                flashcardInfoInstance.setId_word("");
                nextActivity.openActivity(ActivityPanelKits.class);
            }
        });
    }

    @Override
    public void onItemClicked(ModelShowKitsEdit modelShowKitsEdit) {
        flashcardInfoInstance.setId_word(modelShowKitsEdit.getWordID());
        nextActivity.openActivity(ActivityEditFlashcard.class);
    }

   public void showKits() {
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
        Call<FlashcardCollectionsWords> call = jsonFlashcardsCollections.getKit(nameKit);

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
                    Toast.makeText(ActivityShowKitsEdit.this, "Błąd danych", Toast.LENGTH_SHORT).show();
                }
                RefreshRecycleView();
            }

            @Override
            public void onFailure(Call<FlashcardCollectionsWords> call, Throwable t) {
                if(t.getMessage().equals("timeout"))  Toast.makeText(ActivityShowKitsEdit.this,"Uruchamianie serwera", Toast.LENGTH_SHORT).show();
            }
        });
    }


    private void RefreshRecycleView() {
        mRecyclerView = findViewById(R.id.showWordKitsRecycleView);
        mRecyclerView.setHasFixedSize(true);
        mLayoutManager = new LinearLayoutManager(this);
        mAdapter = new AdapterShowKitsEdit(wordsList, this);
        mRecyclerView.setLayoutManager(mLayoutManager);
        mRecyclerView.setAdapter(mAdapter);
    }

    @Override
    public boolean dispatchKeyEvent(KeyEvent event) {
        if (event.getKeyCode() == KeyEvent.KEYCODE_BACK && isBackPressedBlocked) {
            return true;
        }
        return super.dispatchKeyEvent(event);
    }

    private void setID() {
        back = findViewById(R.id.buttonBackShowKits);
    }
}