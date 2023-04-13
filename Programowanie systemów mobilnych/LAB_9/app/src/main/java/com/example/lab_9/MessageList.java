package com.example.lab_9;

public class MessageList {
    private String eName;
    private String eDate;
    private String eMessage;
    private String eID;

    public MessageList(String name, String date, String message, String id){
        eName = name;
        eDate = date;
        eMessage = message;
        eID = id;
    }

    public String geteName() {
        return eName;
    }
    public String geteDate() {
        return eDate;
    }
    public String geteMessage() { return eMessage;}
    public String geteID() { return eID;}
}
