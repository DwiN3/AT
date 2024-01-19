package com.example.Fiszki.security.auth.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a response for token validity check.
 * This class is used to encapsulate the response information indicating whether a given token is valid or not.
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
public class TokenValidityResponse {

    /**
     * A boolean flag indicating the validity of the token.
     * If true, the token is valid; otherwise, it is not valid.
     */
    private boolean access;
}
