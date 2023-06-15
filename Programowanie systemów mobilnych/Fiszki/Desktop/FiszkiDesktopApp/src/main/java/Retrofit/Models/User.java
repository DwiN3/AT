package Retrofit.Models;

import com.google.gson.annotations.SerializedName;

/**
 * Model class representing a registration request
 */
public class User {
    private String nick,email,password, repeatedPassword, token, content;

    @SerializedName("body")
    private String text;

    /**
     * The constructor is responsible for sending a password reminder query
     * @param email            the email
     */
    public User(String email) {
        this.email = email;
    }


    /**
     * The constructor is responsible for sending a login request
     * @param nick          the password
     * @param password      the repeated password
     */
    public User(String nick, String password){
        this.nick = nick;
        this.password = password;
    }

    /**
     * The constructor is responsible for sending a password reset request
     * @param password          the password
     * @param repeatedPassword  the repeated password
     */
    public User(String email, String password, String repeatedPassword) {
        this.email = email;
        this.password = password;
        this.repeatedPassword = repeatedPassword;
    }

    /**
     * The constructor is responsible for sending user registration related queries
     * @param email             the email
     * @param password          the password
     * @param repeatedPassword  the repeated password
     * @param nick              the nickname
     */
    public User(String email, String password, String repeatedPassword, String nick) {
        this.email = email;
        this.password = password;
        this.repeatedPassword = repeatedPassword;
        this.nick = nick;
    }

    /**
     * Returns the nickname
     * @return the nickname
     */
    public String getNick() { return nick; }

    /**
     * Returns the email
     * @return the email
     */
    public String getEmail() { return email; }

    /**
     * Returns the password
     * @return the password
     */
    public String getPassword() { return password; }

    /**
     * Returns the repeated password
     * @return the repeated password
     */
    public String getRepeatedPassword() { return repeatedPassword; }

    /**
     * Returns the content
     * @return the content
     */
    public String getContent() { return content; }

    /**
     * Returns the text
     * @return the text
     */
    public String getText() { return text; }

    /**
     * Returns the token
     * @return the token
     */
    public String getToken() { return token; }
}


