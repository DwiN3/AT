package com.kdbk.fiszki.Retrofit.Models;

import com.google.gson.annotations.SerializedName;

public class UserLVL {


    private int result, level, requiredPoints, points;
    private String content;



    @SerializedName("body")
    private String text;

    public UserLVL(int result) {
        this.result = result;
    }

    public int getResult() {
        return result;
    }

    public String getContent() {
        return content;
    }

    public void setResult(int result) {
        this.result = result;
    }

    public int getLevel() {
        return level;
    }

    public int getRequiredPoints() {
        return requiredPoints;
    }

    public int getPoints() {
        return points;
    }


    public String getText() {
        return text;
    }
}
