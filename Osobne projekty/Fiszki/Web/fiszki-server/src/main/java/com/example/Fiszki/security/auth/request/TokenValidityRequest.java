package com.example.Fiszki.security.auth.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a request for checking the validity of a token.
 * This class is used to encapsulate the information needed to verify the validity of a given token.
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
public class TokenValidityRequest {

    /**
     * The token whose validity needs to be checked.
     */
    String token;
}
