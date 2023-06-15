package com.kdbk.fiszki.Activity;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.os.Bundle;
import android.widget.TextView;
import com.kdbk.fiszki.Instance.GameSettingsInstance;
import com.kdbk.fiszki.RecyclerView.Adaper.AdapterCategories;
import com.kdbk.fiszki.RecyclerView.Model.ModelCategories;
import com.kdbk.fiszki.Other.NextActivity;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.RecyclerView.SelectListener.SelectListenerCategories;
import java.util.ArrayList;

public class ActivityCategories extends AppCompatActivity implements SelectListenerCategories {
    private NextActivity nextActivity = new NextActivity(this);
    private GameSettingsInstance gameSettingsInstance = GameSettingsInstance.getInstance();
    private RecyclerView mRecyclerView;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager mLayoutManager;
    private TextView noCategoriesInfo;
    private String selectedMode = "";
    private ArrayList<ModelCategories> listCategories = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_categories);
        setID();

        listCategories.add(new ModelCategories(R.drawable.category_animal,"zwierzeta", 1));
        listCategories.add(new ModelCategories(R.drawable.category_house,"dom", 2));
        listCategories.add(new ModelCategories(R.drawable.category_shop,"zakupy", 3));
        listCategories.add(new ModelCategories(R.drawable.category_work,"praca",4));
        listCategories.add(new ModelCategories(R.drawable.category_health,"zdrowie",5));
        listCategories.add(new ModelCategories(R.drawable.category_human,"czlowiek",6));
        listCategories.add(new ModelCategories(R.drawable.category_trip,"turystyka",7));
        listCategories.add(new ModelCategories(R.drawable.category_apple,"jedzenie",8));
        listCategories.add(new ModelCategories(R.drawable.category_study,"edukacja",9));
        listCategories.add(new ModelCategories(R.drawable.category_others,"inne",10));

        selectedMode = gameSettingsInstance.getGameMode();

        mRecyclerView = findViewById(R.id.categoriesRecycleView);
        mRecyclerView.setHasFixedSize(true);
        mLayoutManager = new LinearLayoutManager(this);
        mAdapter = new AdapterCategories(listCategories, this);
        mRecyclerView.setLayoutManager(mLayoutManager);
        mRecyclerView.setAdapter(mAdapter);
    }

    @Override
    public void onItemClicked(ModelCategories modelCategories) {
        gameSettingsInstance.setName(modelCategories.getNameCategory());
        gameSettingsInstance.setSelectData("category");

        if(selectedMode.equals("quiz")){
            nextActivity.openActivity(ActivityQuizScreen.class);
        } else if(selectedMode.equals("learn")){
            nextActivity.openActivity(ActivityLearningScreen.class);
        }
    }
    private void setID() {
        noCategoriesInfo = findViewById(R.id.textNoCategories);
    }
}