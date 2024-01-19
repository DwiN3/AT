package com.example.Fiszki.security.auth;

import com.example.Fiszki.Instance.TokenInstance;
import com.example.Fiszki.security.auth.request.*;
import com.example.Fiszki.security.auth.response.TokenValidityResponse;
import com.example.Fiszki.security.auth.response.UserDateResponse;
import com.example.Fiszki.security.auth.response.UserInfoResponse;
import com.example.Fiszki.security.auth.response.UserLevelResponse;
import com.example.Fiszki.security.config.JwtService;
import com.example.Fiszki.security.user.Role;
import com.example.Fiszki.security.user.User;
import com.example.Fiszki.security.user.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

/**
 * Service class handling user authentication and related operations.
 */
@Service
@RequiredArgsConstructor
public class AuthenticationService {
    private TokenInstance tokenInstance = TokenInstance.getInstance();
    private final UserRepository repository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;
    private final AuthenticationManager authenticationManager;

    /**
     * Registers a new user with the provided registration request.
     *
     * @param request The registration request containing user information.
     * @return UserInfoResponse containing the registration response.
     */
    public UserInfoResponse register(RegisterRequest request) {
        if (!isValidRegistrationRequest(request)) {
            return UserInfoResponse.builder().response("Invalid registration request.").build();
        }

        if (!request.getPassword().equals(request.getRepeatedPassword())){
            return UserInfoResponse.builder().response("Repeated password is different from the password.").build();
        }

        if (repository.findByEmail(request.getEmail()).isPresent()) {
            return UserInfoResponse.builder().response("User with given e-mail already exists.").build();
        }

        var user = User.builder()
                .level(1)
                .points(0)
                .firstName(request.getFirstname())
                .lastName(request.getLastname())
                .email(request.getEmail())
                .password(passwordEncoder.encode(request.getPassword()))
                .role(Role.USER)
                .build();

        repository.save(user);
        var jwtToken = jwtService.generateToken(user);
        return UserInfoResponse.builder().response("User added successfully.").build();
    }

    /**
     * Validates a registration request for completeness and correctness.
     *
     * @param request The registration request to be validated.
     * @return True if the registration request is valid, false otherwise.
     */
    private boolean isValidRegistrationRequest(RegisterRequest request) {
        if (request.getFirstname().isEmpty() || request.getLastname().isEmpty() ||
                request.getEmail().isEmpty() || request.getPassword().isEmpty()) {
            return false;
        }

        if (!isValidEmail(request.getEmail())) {
            return false;
        }

        return request.getPassword().length() >= 5;
    }

    /**
     * Validates an email address based on a simple regex pattern.
     *
     * @param email The email address to be validated.
     * @return True if the email address is valid, false otherwise.
     */
    private boolean isValidEmail(String email) {
        return email != null && email.matches("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}");
    }

    /**
     * Authenticates a user with the provided authentication request.
     *
     * @param request The authentication request containing user credentials.
     * @return UserInfoResponse containing the authentication response.
     */
    public UserInfoResponse authenticate(AuthenticationRequest request) {
        // Verify the existence of a user by email address in the database.
        var userOptional = repository.findByEmail(request.getEmail());
        if (userOptional.isPresent()) {
            authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(
                            request.getEmail(),
                            request.getPassword()
                    )
            );

            var user = repository.findByEmail(request.getEmail()).orElseThrow();
            var jwtToken = jwtService.generateToken(user);

            tokenInstance.setToken(request.getEmail());
            tokenInstance.setUserName(user.getUsername());
            return UserInfoResponse.builder().response(jwtToken).build();
        } else {
            return null;
        }
    }

    /**
     * Retrieves user level information.
     *
     * @return UserLevelResponse containing the user level information.
     */
    public UserLevelResponse userLevel() {
        var userEmail = TokenInstance.getInstance().getToken();
        var user = repository.findByEmail(userEmail).orElseThrow();
        int nextLVLPoints = calculateNextLVLPoints(user.getPoints(), user.getLevel());
        return UserLevelResponse.builder()
                .level(user.getLevel())
                .points(user.getPoints())
                .nextLVLPoints(nextLVLPoints)
                .build();
    }

    /**
     * Updates user points and calculates the next level points.
     *
     * @param pointsRequest The request containing the points to be added.
     * @return UserLevelResponse containing the updated user level information.
     */
    public UserLevelResponse sendPoints(PointsRequest pointsRequest) {
        var userEmail = tokenInstance.getToken();
        var user = repository.findByEmail(userEmail).orElseThrow();

        user.setPoints(user.getPoints() + pointsRequest.getPoints());
        updateLevel(user);

        repository.save(user);

        int nextLVLPoints = calculateNextLVLPoints(user.getPoints(), user.getLevel());

        return UserLevelResponse.builder()
                .level(user.getLevel())
                .points(user.getPoints())
                .nextLVLPoints(nextLVLPoints)
                .build();
    }

    /**
     * Calculates the total points required to reach the next level.
     *
     * @param currentPoints The current points of the user.
     * @param currentLevel  The current level of the user.
     * @return The total points required to reach the next level.
     */
    private int calculateNextLVLPoints(int currentPoints, int currentLevel) {
        int requiredPoints = calculateRequiredPoints(currentLevel);
        return (currentLevel + 1) * requiredPoints;
    }

    /**
     * Calculates the required points for a specific level.
     *
     * @param currentLevel The current level for which points are calculated.
     * @return The required points for the given level.
     */
    private int calculateRequiredPoints(int currentLevel) {
        return 0 + (currentLevel * 50);
    }

    /**
     * Updates the user's level based on the points they have earned.
     *
     * @param user The user for whom the level needs to be updated.
     */
    private void updateLevel(User user) {
        int requiredPoints = calculateRequiredPoints(user.getLevel());
        int nextLVLPoints = (user.getLevel() + 1) * requiredPoints;

        if (user.getPoints() >= nextLVLPoints) {
            user.setLevel(user.getLevel() + 1);
        }
    }

    /**
     * Retrieves user information.
     *
     * @return UserDateResponse containing the user information.
     */
    public UserDateResponse getInfo() {
        var userEmail = TokenInstance.getInstance().getToken();
        var user = repository.findByEmail(userEmail).orElseThrow();
        return UserDateResponse.builder()
                .id(user.getId())
                .firstName(user.getFirstName())
                .lastName(user.getLastName())
                .email(user.getEmail())
                .points(user.getPoints())
                .level(user.getLevel())
                .build();
    }

    /**
     * Changes the user's password.
     *
     * @param request The request containing old and new password information.
     * @return UserInfoResponse containing the password change response.
     */
    public UserInfoResponse changePassword(ChangePasswordRequest request) {
        var userEmail = tokenInstance.getToken();

        var optionalUser = repository.findByEmail(request.getEmail());

        if (optionalUser.isEmpty()) {
            System.out.println(optionalUser);
            return UserInfoResponse.builder().response("User with given e-mail does not exist.").build();
        }

        if (request.getNew_password().isEmpty()) {
            return UserInfoResponse.builder().response("New password cannot be empty.").build();
        }

        var user = optionalUser.get();

        if (!passwordEncoder.matches(request.getPassword(), user.getPassword())) {
            return UserInfoResponse.builder().response("Current password is incorrect.").build();
        }

        if (request.getNew_password().equals(request.getPassword())) {
            return UserInfoResponse.builder().response("New password must be different from the current password.").build();
        }

        if (request.getNew_password().length() < 5) {
            return UserInfoResponse.builder().response("New password must be at least 5 characters long.").build();
        }

        if (!request.getNew_password().equals(request.getRe_new_password())) {
            return UserInfoResponse.builder().response("New passwords do not match.").build();
        }

        user.setPassword(passwordEncoder.encode(request.getNew_password()));
        repository.save(user);

        return UserInfoResponse.builder().response("Password changed successfully.").build();
    }

    /**
     * Changes the user's password using a link.
     *
     * @param request The request containing email and new password information.
     * @return UserInfoResponse containing the password change response.
     */
    public UserInfoResponse changePasswordLink(ChangePasswordFromLinkRequest request) {
        // Verify the existence of a user by email address in the database.
        var optionalUser = repository.findByEmail(request.getEmail());

        if (optionalUser.isEmpty()) {
            return UserInfoResponse.builder().response("User with given e-mail does not exist.").build();
        }

        var user = optionalUser.get();

        // Validate the new password fields
        if (request.getNew_password().isEmpty() || request.getNew_password().length() < 5) {
            return UserInfoResponse.builder().response("New password must be at least 5 characters long.").build();
        }

        if (!request.getNew_password().equals(request.getRe_new_password())) {
            return UserInfoResponse.builder().response("New passwords do not match.").build();
        }

        // Update the user's password
        user.setPassword(passwordEncoder.encode(request.getNew_password()));
        repository.save(user);

        return UserInfoResponse.builder().response("Password changed successfully.").build();
    }

    /**
     * Deletes a user account.
     *
     * @param userPassword The password of the user to be deleted.
     * @return UserInfoResponse containing the user deletion response.
     */
    public UserInfoResponse deleteUser(String userPassword) {
        var userEmail = tokenInstance.getToken();
        var user = repository.findByEmail(userEmail).orElseThrow();

        if (!passwordEncoder.matches(userPassword, user.getPassword())) {
            return UserInfoResponse.builder().response("Incorrect password. User not deleted.").build();
        }

        repository.delete(user);
        return UserInfoResponse.builder().response("User deleted successfully.").build();
    }

    /**
     * Checks the validity of a token and grants access accordingly.
     *
     * @param tokenValidityRequest The request containing the token to be checked.
     * @return TokenValidityResponse containing the token validity response.
     */
    public TokenValidityResponse checkAccess(TokenValidityRequest tokenValidityRequest) {
        String token = tokenValidityRequest.getToken();

        boolean isValidToken = jwtService.validateToken(token);

        return TokenValidityResponse.builder().access(isValidToken).build();
    }
}
