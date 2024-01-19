package com.example.Fiszki.flashcards.flashcard;

import org.springframework.http.HttpStatus;
import java.time.LocalDateTime;

/**
 * Represents an API error with details such as status, timestamp, error message, and request path.
 */
public class ApiError {

    private int status;
    private LocalDateTime timestamp;
    private String message;
    private String path;

    /**
     * Constructs a new ApiError instance with the specified HTTP status, error message, and request path.
     *
     * @param status   The HTTP status code of the error.
     * @param message  A description of the error.
     * @param path     The path of the request that caused the error.
     */
    public ApiError(HttpStatus status, String message, String path) {
        this.status = status.value();
        this.timestamp = LocalDateTime.now();
        this.message = message;
        this.path = path;
    }

    /**
     * Gets the HTTP status code of the error.
     *
     * @return The HTTP status code.
     */
    public int getStatus() {
        return status;
    }

    /**
     * Gets the timestamp when the error occurred.
     *
     * @return The timestamp of the error.
     */
    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    /**
     * Gets the error message.
     *
     * @return A description of the error.
     */
    public String getMessage() {
        return message;
    }

    /**
     * Gets the path of the request that caused the error.
     *
     * @return The path of the request.
     */
    public String getPath() {
        return path;
    }
}