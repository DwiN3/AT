package com.kdbk.fiszki.Retrofit.Models;

import com.google.gson.annotations.SerializedName;

public class Register {
    private String nick,email,password, repeatedPassword, content;

    @SerializedName("body")
    private String text;

    public Register(String email, String password, String repeatedPassword, String nick) {
        this.email = email;
        this.password = password;
        this.repeatedPassword = repeatedPassword;
        this.nick = nick;
    }

    public Register(String email, String password, String repeatedPassword) {
        this.email = email;
        this.password = password;
        this.repeatedPassword = repeatedPassword;
    }

    public String getNick() {
        return nick;
    }

    public String getEmail() {
        return email;
    }

    public String getPassword() {
        return password;
    }

    public String getRepeatedPassword() {
        return repeatedPassword;
    }

    public String getContent() {
        return content;
    }

    public String getText() {
        return text;
    }
}

