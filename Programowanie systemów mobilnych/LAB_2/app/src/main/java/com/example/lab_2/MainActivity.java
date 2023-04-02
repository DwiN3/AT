package com.example.lab_2;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {
    EditText number;
    Button accept;
    RadioGroup buttons;
    TextView show_result;
    Button cm, km, mile, ly;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        number = findViewById(R.id.editTextNumber);
        accept = findViewById(R.id.button);
        buttons = findViewById(R.id.radioGroup);
        show_result = findViewById(R.id.textView);

        cm = findViewById(R.id.radio_one);
        km = findViewById(R.id.radio_two);
        mile = findViewById(R.id.radio_three);
        ly = findViewById(R.id.radio_four);

        accept.setOnClickListener(new View.OnClickListener() {
            @Override

            public void onClick(View view) {
                double multiplier = 1;
                int radioID = buttons.getCheckedRadioButtonId();

                if(radioID == cm.getId()) multiplier = 10;
                else if(radioID == km.getId()) multiplier = 0.001;
                else if(radioID == mile.getId()) multiplier = 0.000621371192;
                else if(radioID == ly.getId()) multiplier = 1.057000834024;

                String number_in_app = number.getText().toString();
                double value = Integer.parseInt(number_in_app);
                double result = value * multiplier;
                show_result.setText("Wynik = " + result);
            }
        });
    }
}