package com.example.Fiszki.security.user;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.repository.query.Param;
import java.util.Optional;

/**
 * Repository interface for managing user data in the database.
 */
public interface UserRepository extends JpaRepository<User, Integer> {

    /**
     * Find a user by email address.
     *
     * @param email The email address of the user.
     * @return An {@link Optional} containing the user, or empty if not found.
     */
    Optional<User> findByEmail(@Param("email")String email);

}
