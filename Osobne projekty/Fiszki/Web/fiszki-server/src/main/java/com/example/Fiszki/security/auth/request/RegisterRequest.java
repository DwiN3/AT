package com.example.Fiszki.security.auth.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a request for user registration.
 * This class is used to encapsulate the information required for user registration,
 * including the user's first name, last name, email address, password, and the repeated password for confirmation.
 *
 * The class is annotated with Lombok annotations to automatically generate common boilerplate code:
 * - @Data: Generates getters, setters, toString, equals, and hashCode methods.
 * - @Builder: Provides a builder pattern for easy object creation.
 * - @AllArgsConstructor: Generates an all-args constructor.
 * - @NoArgsConstructor: Generates a no-args constructor.
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class RegisterRequest {

    /**
     * The user's first name.
     */
    private String firstname;

    /**
     * The user's last name.
     */
    private String lastname;

    /**
     * The email address for user registration.
     */
    private String email;

    /**
     * The password for user registration.
     */
    private String password;

    /**
     * The repeated password for confirmation during user registration.
     */
    private String repeatedPassword;
}
