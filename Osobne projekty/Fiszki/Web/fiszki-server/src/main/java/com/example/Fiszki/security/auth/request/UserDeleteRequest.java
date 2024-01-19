package com.example.Fiszki.security.auth.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a request for deleting a user account.
 * This class is used to encapsulate the information required for user account deletion,
 * specifically the user's password for authentication.
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
public class UserDeleteRequest {

    /**
     * The user's password for authentication before account deletion.
     */
    private String password;
}
