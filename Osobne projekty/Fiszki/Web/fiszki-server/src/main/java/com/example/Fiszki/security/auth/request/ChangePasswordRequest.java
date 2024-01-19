package com.example.Fiszki.security.auth.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a request for changing the user password.
 * This class is used to encapsulate the necessary information for changing the password,
 * including the user's email, the current password, the new password, and the confirmation of the new password.
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
public class ChangePasswordRequest {

    /**
     * The email address associated with the user account.
     */
    private String email;

    /**
     * The current password associated with the user account.
     */

    private String password;

    /**
     * The new password to be set for the user account.
     */
    private String new_password;

    /**
     * The confirmation of the new password to ensure correctness.
     */
    private String re_new_password;
}
