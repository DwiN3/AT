package com.example.calculator_35196;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import org.mariuszgromada.math.mxparser.*;

public class MainActivity extends AppCompatActivity {
    Button b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,bAC,bp_s,bproc,bdot,bdiv,blog10,bmul,bfac,bmin,bsqrt,bplus,bpow3,bqual,bpow2;
    TextView screenMain, screenSec;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setID();

        b0.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"0"); }
        });
        b1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {screenMain.setText(screenMain.getText()+"1"); }
        });
        b2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"2"); }
        });
        b3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"3"); }
        });
        b4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"4"); }
        });
        b5.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"5"); }
        });
        b6.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"6"); }
        });
        b7.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"7"); }
        });
        b8.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"8"); }
        });
        b9.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"9"); }
        });
        bdot.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"."); }
        });
        bplus.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"+"); }
        });
        bmin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"-"); }
        });
        bmul.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(screenMain.getText()+"×"); }
        });
        bdiv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) { screenMain.setText(screenMain.getText()+"÷"); }
        });
        bAC.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { screenMain.setText(""); screenSec.setText(""); }
        });
        bp_s.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String val_string = screenMain.getText().toString();

                if(checkLetters(val_string,true)){
                    double d = Double.parseDouble(val_string);
                    double res = d*(-1);
                    screenMain.setText(String.valueOf(res));
                    screenSec.setText(String.valueOf(res*-1));
                }
                else { errorScreen(); }
            }
        });
        bproc.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String val_string = screenMain.getText().toString();

                if(checkLetters(val_string,true)) {
                    double d = Double.parseDouble(val_string);
                    double res = d * 0.01;
                    screenMain.setText(String.valueOf(res));
                    screenSec.setText(d + "% = ");
                }
                else { errorScreen(); }
            }
        });
        blog10.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String val_string = screenMain.getText().toString();

                if(checkLetters(val_string,true)) {
                    double d = Double.parseDouble(val_string);
                    double res = Math.log10(d);
                    screenMain.setText(String.valueOf(res));
                    screenSec.setText("log₁₀"+d+" = ");
                }
                else { errorScreen(); }
            }
        });
        bfac.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String val_string = screenMain.getText().toString();

                if(checkLetters(val_string,true)) {
                    double val = Integer.parseInt(val_string);
                    double fact = factorial(val);
                    screenMain.setText(String.valueOf(fact));
                    screenSec.setText(val + "! = ");
                }
                else { errorScreen(); }
            }
        });
        bsqrt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String val_string = screenMain.getText().toString();

                if(checkLetters(val_string, true)) {
                    double d = Math.sqrt(Double.parseDouble(val_string));
                    screenMain.setText(String.valueOf(d));
                    screenSec.setText("SQRT("+val_string+") = ");
                }
                else { errorScreen(); }
            }
        });
        bpow2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String val_string = screenMain.getText().toString();

                if(checkLetters(val_string, true)) {
                    double d = Double.parseDouble(val_string);
                    double square = d * d;
                    screenMain.setText(String.valueOf(square));
                    screenSec.setText(d + "² =");
                }
                else { errorScreen(); }
            }
        });
        bpow3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String val_string = screenMain.getText().toString();

                if(checkLetters(val_string, true)) {
                    double d = Double.parseDouble(val_string);
                    double square = d*d*d;
                    screenMain.setText(String.valueOf(square));
                    screenSec.setText(d+"³ =");
                }
                else { errorScreen(); }
            }
        });
        bqual.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String val_string = screenMain.getText().toString();

                if(checkLetters(val_string, false)){
                    String replaced_string = val_string.replace('÷','/').replace('×','*');
                    Expression res = new Expression(replaced_string);
                    String result = String.valueOf(res.calculate());
                    screenMain.setText(result);
                    screenSec.setText(val_string);
                }
                else { errorScreen(); }
            }
        });
    }

    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState){
        super.onSaveInstanceState(outState);
        outState.putString("MAIN", screenMain.getText().toString());
        outState.putString("SEC", screenSec.getText().toString());
    }

    @Override
    protected void onRestoreInstanceState(@NonNull Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        String main = savedInstanceState.getString("MAIN", "0");
        String sec = savedInstanceState.getString("SEC", "0");
        screenMain.setText(main);
        screenSec.setText(sec);
    }

    double factorial(double n) {
        double factorial = 1;
        for(double i = n ; i>=1 ;i--) factorial = factorial*i;
        return factorial;
    }

    void errorScreen(){
        screenMain.setText("");
        screenSec.setText("Error");
    }

    boolean checkLetters(String val_string, boolean moreOpcions){
        if(!val_string.isEmpty()){
            char first_letter = val_string.charAt(0);
            char last_letter = val_string.charAt(val_string.length()-1);
            for(int n=0;n < val_string.length()-1;n++){
                if((moreOpcions == true ) && ((val_string.charAt(n) == '÷' || val_string.charAt(n) == '×' || val_string.charAt(n) == '+' || val_string.charAt(n+1) == '-'))) return false;
                if((val_string.charAt(n) == '÷' || val_string.charAt(n) == '×' || val_string.charAt(n) == '.' || val_string.charAt(n) == '+' ) && (val_string.charAt(n+1) == '÷' || val_string.charAt(n+1) == '×' || val_string.charAt(n+1) == '.' || val_string.charAt(n+1) == '+')) return false;
            }
            if(((first_letter >= '0' && first_letter <= '9') || first_letter == '-') && (last_letter >= '0' && last_letter <= '9')) return true;
        }
        return false;
    }

    void setID(){
        screenMain = findViewById(R.id.id_screenmain);
        screenSec = findViewById(R.id.id_screensec);
        b0 = findViewById(R.id.id_0);
        b1 = findViewById(R.id.id_1);
        b2 = findViewById(R.id.id_2);
        b3 = findViewById(R.id.id_3);
        b4 = findViewById(R.id.id_4);
        b5 = findViewById(R.id.id_5);
        b6 = findViewById(R.id.id_6);
        b7 = findViewById(R.id.id_7);
        b8 = findViewById(R.id.id_8);
        b9 = findViewById(R.id.id_9);
        bAC = findViewById(R.id.id_AC);
        bp_s = findViewById(R.id.id_p_s);
        bproc = findViewById(R.id.id_proc);
        bdot = findViewById(R.id.id_dot123);
        bdiv = findViewById(R.id.id_division);
        blog10 = findViewById(R.id.id_log);
        bmul = findViewById(R.id.id_multiplication);
        bfac = findViewById(R.id.id_x);
        bmin = findViewById(R.id.id_minus);
        bsqrt = findViewById(R.id.id_SQRT);
        bplus = findViewById(R.id.id_plus);
        bpow3 = findViewById(R.id.id_might_three);
        bqual = findViewById(R.id.id_qual);
        bpow2 = findViewById(R.id.id_might_two);
    }
}