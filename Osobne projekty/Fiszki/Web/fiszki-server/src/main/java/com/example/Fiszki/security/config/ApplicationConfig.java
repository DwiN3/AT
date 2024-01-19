package com.example.Fiszki.security.config;

import com.example.Fiszki.security.user.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

/**
 * Configuration class responsible for providing essential security beans.
 * It configures a custom UserDetailsService, AuthenticationProvider, AuthenticationManager, and PasswordEncoder.
 * The UserDetailsService retrieves user details from the repository based on the email.
 * The AuthenticationProvider uses the custom UserDetailsService and a BCryptPasswordEncoder for password hashing.
 * The AuthenticationManager is obtained from the AuthenticationConfiguration.
 * The PasswordEncoder utilizes BCryptPasswordEncoder for secure password hashing.
 */
@Configuration
@RequiredArgsConstructor
public class ApplicationConfig {

    private final UserRepository repository;

    /**
     * Provides an implementation of the UserDetailsService interface.
     * Retrieves user details from the repository based on the email.
     *
     * @return UserDetailsService implementation
     */
    @Bean
    public UserDetailsService userDetailsService() {
        return username -> repository.findByEmail(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found!"));
    }

    /**
     * Configures and provides an AuthenticationProvider implementation.
     * Uses the custom UserDetailsService and a BCryptPasswordEncoder for password hashing.
     *
     * @return AuthenticationProvider implementation
     */
    @Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider authenticationProvider = new DaoAuthenticationProvider();
        authenticationProvider.setUserDetailsService(userDetailsService());
        authenticationProvider.setPasswordEncoder(passwordEncoder());

        return authenticationProvider;
    }

    /**
     * Provides an implementation of the AuthenticationManager interface.
     *
     * @param configuration AuthenticationConfiguration used to obtain the authentication manager.
     * @return AuthenticationManager implementation
     * @throws Exception if an error occurs during authentication manager creation.
     */
    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration configuration) throws Exception {
        return configuration.getAuthenticationManager();
    }

    /**
     * Provides an implementation of the PasswordEncoder interface.
     * Uses BCryptPasswordEncoder for secure password hashing.
     *
     * @return PasswordEncoder implementation
     */
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

}
