package com.example.lab_1;

import static android.view.View.INVISIBLE;
import static android.view.View.VISIBLE;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void nameTextView(View view) {
        TextView myName = findViewById(R.id.nameTextView);

        if(myName.getVisibility() == VISIBLE) myName.setVisibility(INVISIBLE);
        else myName.setVisibility(VISIBLE);
    }
}