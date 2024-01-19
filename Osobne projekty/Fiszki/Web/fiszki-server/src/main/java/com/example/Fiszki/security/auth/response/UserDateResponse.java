package com.example.Fiszki.security.auth.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a response containing user information.
 * This class is used to encapsulate information about a user, including their id, first name, last name,
 * email address, points, and level.
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
public class UserDateResponse {

    /**
     * The unique identifier for the user.
     */
    private Integer id;

    /**
     * The user's first name.
     */
    private String firstName;

    /**
     * The user's last name.
     */
    private String lastName;

    /**
     * The email address associated with the user.
     */
    private String email;

    /**
     * The number of points accumulated by the user.
     */
    private int points;

    /**
     * The level of the user based on their points.
     */
    private int level;
}
