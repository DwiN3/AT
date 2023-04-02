package com.example.lab_3;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.util.Random;

public class MainActivity extends AppCompatActivity {
    TextView joke_text;
    TextView j1,j2,j3,j4,j5,j6,j7,j8,j9,j10;
    Button button;
    int last_number = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setID();

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Random random = new Random();
                int num = random.nextInt(10);

                if(num == 1 && num != last_number) joke_text.setText(j1.getText());
                else if(num == 2 && num != last_number) joke_text.setText(j2.getText());
                else if(num == 3 && num != last_number) joke_text.setText(j3.getText());
                else if(num == 4 && num != last_number) joke_text.setText(j4.getText());
                else if(num == 5 && num != last_number) joke_text.setText(j5.getText());
                else if(num == 6 && num != last_number) joke_text.setText(j6.getText());
                else if(num == 7 && num != last_number) joke_text.setText(j7.getText());
                else if(num == 8 && num != last_number) joke_text.setText(j8.getText());
                else if(num == 9 && num != last_number) joke_text.setText(j9.getText());
                else if(num == 10 && num != last_number) joke_text.setText(j10.getText());
                last_number = num;
            }
        });
    }

    void setID(){
        joke_text = (TextView) findViewById(R.id.id_textView);
        button = (Button) findViewById(R.id.id_button);
        j1 = (TextView) findViewById(R.id.joke_1);
        j2 = (TextView) findViewById(R.id.joke_2);
        j3 = (TextView) findViewById(R.id.joke_3);
        j4 = (TextView) findViewById(R.id.joke_4);
        j5 = (TextView) findViewById(R.id.joke_5);
        j6 = (TextView) findViewById(R.id.joke_6);
        j7 = (TextView) findViewById(R.id.joke_7);
        j8 = (TextView) findViewById(R.id.joke_8);
        j9 = (TextView) findViewById(R.id.joke_9);
        j10 = (TextView)findViewById(R.id.joke_10);
    }
}