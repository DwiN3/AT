package com.example.Fiszki.security.auth.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a response containing user level information.
 * This class is used to encapsulate information about the user's current level, accumulated points, and the points needed
 * to reach the next level.
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
public class UserLevelResponse {

    /**
     * The current level of the user.
     */
    private int level;

    /**
     * The total number of points accumulated by the user.
     */
    private int points;

    /**
     * The number of points needed to reach the next level.
     */
    private int nextLVLPoints;
}
