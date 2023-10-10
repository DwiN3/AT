package com.kdbk.fiszki.Retrofit.Models;

import com.google.gson.annotations.SerializedName;

public class Login {
    private String nick, password, content, token;

    @SerializedName("body")
    private String text;

    public Login(String nick, String password){
        this.nick = nick;
        this.password = password;
    }

    public String getToken() { return token;}
    public String getNick() {
        return nick;
    }

    public String getPassword() {
        return password;
    }

    public String getContent() {
        return content;
    }

    public String getText() {
        return text;
    }
}
