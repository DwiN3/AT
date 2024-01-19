package com.example.Fiszki.Instance;

/**
 * Singleton class representing an instance of user authentication token information.
 * This class is responsible for managing and providing access to the user's username,
 * authentication token, and server status.
 */
public class TokenInstance {

    /**
     * Singleton instance of the TokenInstance class.
     */
    private static TokenInstance instance = null;

    /**
     * User's username associated with the authentication token.
     */
    private static String userName = "";

    /**
     * Authentication token associated with the user.
     */
    private static String token = "";

    /**
     * Flag indicating the status of the server.
     */
    private static boolean serverStatus = false;

    /**
     * Retrieves the singleton instance of TokenInstance.
     *
     * @return The singleton instance of TokenInstance.
     */
    public static TokenInstance getInstance() {
        if (instance == null) {
            instance = new TokenInstance();
        }
        return instance;
    }

    /**
     * Retrieves the username associated with the authentication token.
     *
     * @return The user's username.
     */
    public String getUserName() {
        return userName;
    }

    /**
     * Retrieves the authentication token.
     *
     * @return The authentication token.
     */
    public String getToken() {
        return token;
    }

    /**
     * Sets the username associated with the authentication token.
     *
     * @param userName The user's username.
     */
    public void setUserName(String userName) {
        TokenInstance.userName = userName;
    }

    /**
     * Sets the authentication token.
     *
     * @param token The authentication token.
     */
    public void setToken(String token) {
        TokenInstance.token = token;
    }

    /**
     * Retrieves the status of the server.
     *
     * @return True if the server is online, false otherwise.
     */
    public boolean isServerStatus() {
        return serverStatus;
    }

    /**
     * Sets the status of the server.
     *
     * @param serverStatus True if the server is online, false otherwise.
     */
    public void setServerStatus(boolean serverStatus) {
        TokenInstance.serverStatus = serverStatus;
    }
}