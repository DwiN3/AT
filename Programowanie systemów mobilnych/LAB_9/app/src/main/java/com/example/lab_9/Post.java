package com.example.lab_9;

import com.google.gson.annotations.SerializedName;

public class Post {
    private String login, date, id, content;

    @SerializedName("body")
    private String text;

    public Post(String login, String content) {
        this.login = login;
        this.content = content;
    }

    public String getLogin() {
        return login;
    }

    public String getDate() {
        return date;
    }

    public String getId() {return id; }

    public String getContent() {
        return content;
    }

}
