package com.example.calculator_35196;

public class Calculations
{
    int pos = -1;
    int sign;
    String str;

    void nextChar() {
        if(++pos < str.length()) sign = str.charAt(pos);
        else sign = -1;
    }

    boolean take(int charToTake) {
        if (sign == charToTake) {
            nextChar();
            return true;
        }
        return false;
    }

    double parse(String strOuter) {
        str = strOuter;
        nextChar();
        double x = parseTerm();
        return x;
    }

    double parseTerm() {
        double x = parseFactor();
        while(true){
            if      (take('*')) x *= parseFactor();
            else if (take('/')) x /= parseFactor();
            else if (take('+')) x += parseTerm();
            else if (take('-')) x -= parseTerm();
            else return x;
        }
    }

    double parseFactor() {
        if (take('+')) return parseFactor();
        if (take('-')) return -parseFactor();
        double val;
        int startPos = pos;
        while ((sign >= '0' && sign <= '9') || sign == '.') nextChar();
        val = Double.parseDouble(str.substring(startPos, pos));
        return val;
    }
}
