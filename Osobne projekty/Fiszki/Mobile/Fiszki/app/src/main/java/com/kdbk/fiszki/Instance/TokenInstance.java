package com.kdbk.fiszki.Instance;

public class TokenInstance {
    private static TokenInstance instance = null;
    private static String userName ="", token="";
    private static boolean serverStatus= false;

    public static TokenInstance getInstance() {
        if (instance == null) {
            instance = new TokenInstance();
        }
        return instance;
    }

    public String getUserName() {
        return userName;
    }

    public String getToken() {
        return token;
    }

    public void setUserName(String userName) {
        TokenInstance.userName = userName;
    }

    public void setToken(String token) {
        TokenInstance.token = token;
    }

    public boolean isServerStatus() {
        return serverStatus;
    }

    public void setServerStatus(boolean serverStatus) {
        TokenInstance.serverStatus = serverStatus;
    }
}
