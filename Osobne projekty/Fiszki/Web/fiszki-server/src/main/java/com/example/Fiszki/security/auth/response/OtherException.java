package com.example.Fiszki.security.auth.response;


/**
 * Custom exception class for handling other runtime exceptions.
 * This class extends the RuntimeException class and is used to represent exceptions specific to the application
 * that do not fall into a predefined category.
 */
public class OtherException extends RuntimeException {

    /**
     * Constructs a new OtherException with the specified detail message.
     *
     * @param message The detail message of the exception.
     */
    public OtherException(String message) {
        super(message);
    }
}
