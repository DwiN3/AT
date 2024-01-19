package com.example.Fiszki.security.auth;

import com.example.Fiszki.security.auth.request.*;
import com.example.Fiszki.security.auth.response.*;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.web.bind.annotation.*;
import java.util.NoSuchElementException;

/**
 * Controller class handling authentication-related endpoints.
 */
@RestController
@RequestMapping("/flashcards/auth/")
@RequiredArgsConstructor
public class AuthenticationController {
    private final AuthenticationService authenticationService;

    /**
     * Endpoint for user registration.
     *
     * @param request The registration request containing user information.
     * @return ResponseEntity containing the registration response.
     */
    @PostMapping("/register")
    public ResponseEntity<UserInfoResponse> register (@RequestBody RegisterRequest request) {
        UserInfoResponse response = authenticationService.register(request);
        if (response.getResponse().equals("Invalid registration request.")) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
        } else if (response.getResponse().equals("User with given e-mail already exists.")) {
            return ResponseEntity.status(HttpStatus.CONFLICT).body(response);
        } else if (response.getResponse().equals("Repeated password is different from the password.")) {
            return ResponseEntity.status(HttpStatus.CONFLICT).body(response);
        }
        return ResponseEntity.ok(response);
    }

    /**
     * Endpoint for user authentication (login).
     *
     * @param authenticate The authentication request containing user credentials.
     * @return ResponseEntity containing the authentication response.
     */
    @PostMapping("/login")
    public ResponseEntity<?> authenticate(@RequestBody AuthenticationRequest authenticate) {
        try {
            UserInfoResponse response = authenticationService.authenticate(authenticate);
            if (response == null) {
                // Zwrócenie informacji o braku użytkownika z podanym e-mail w bazie
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(UserInfoResponse.builder().response("User with given e-mail does not exist.").build());
            }
            return ResponseEntity.ok(response);
        } catch (AuthenticationException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(UserInfoResponse.builder().response("Authentication error: " + e.getMessage()).build());
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(UserInfoResponse.builder().response("Other error: " + e.getMessage()).build());
        }
    }

    /**
     * Endpoint for retrieving user level information.
     *
     * @return ResponseEntity containing the user level response.
     */
    @GetMapping("/level")
    public ResponseEntity<?> userLevel() {
        try {
            UserLevelResponse userLevelResponse = authenticationService.userLevel();
            return ResponseEntity.ok(userLevelResponse);
        } catch (AccessDeniedException e) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body(UserInfoResponse.builder().response("Access denied: " + e.getMessage()).build());
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(UserInfoResponse.builder().response("Other error: " + e.getMessage()).build());
        } catch (NoSuchElementException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(UserInfoResponse.builder().response("NoSuchElementException: " + "Invalid token").build());
        }
    }

    /**
     * Endpoint for sending user points.
     *
     * @param pointsRequest The request containing the points to be sent.
     * @return ResponseEntity containing the user level response.
     */
    @PutMapping("/points")
    public ResponseEntity<?> sendPoints(@RequestBody PointsRequest pointsRequest) {
        try {
            UserLevelResponse userLevelResponse = authenticationService.sendPoints(pointsRequest);
            return ResponseEntity.ok(userLevelResponse);
        } catch (AccessDeniedException e) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body(UserInfoResponse.builder().response("Access denied: " + e.getMessage()).build());
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(UserInfoResponse.builder().response("Other error: " + e.getMessage()).build());
        } catch (NoSuchElementException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(UserInfoResponse.builder().response("NoSuchElementException: " + "Invalid token").build());
        }
    }

    /**
     * Endpoint for retrieving user information.
     *
     * @return ResponseEntity containing the user information response.
     */
    @GetMapping("/info")
    public ResponseEntity<?> getUserInfo() {
        try {
            UserDateResponse userLevelResponse = authenticationService.getInfo();
            return ResponseEntity.ok(userLevelResponse);
        } catch (AccessDeniedException e) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body(UserInfoResponse.builder().response("Access denied: " + e.getMessage()).build());
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(UserInfoResponse.builder().response("Other error: " + e.getMessage()).build());
        } catch (NoSuchElementException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(UserInfoResponse.builder().response("NoSuchElementException: " + "Invalid token").build());
        }
    }

    /**
     * Endpoint for changing user password.
     *
     * @param request The request containing old and new password information.
     * @return ResponseEntity containing the password change response.
     */
    @PutMapping("/change-password")
    public ResponseEntity<?> changePassword(@RequestBody ChangePasswordRequest request) {
        try {
            UserInfoResponse response = authenticationService.changePassword(request);
            if (response.getResponse().equals("User with given e-mail does not exist.")) {
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
            } else if (!response.getResponse().equals("Password changed successfully.")) {
                return ResponseEntity.status(HttpStatus.CONFLICT).body(response);
            }
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(UserInfoResponse.builder().response("Other error: " + e.getMessage()).build());
        }
    }

    /**
     * Endpoint for processing user password change from a link.
     *
     * @param request The request containing email and new password information.
     * @return ResponseEntity containing the password change response.
     */
    @PutMapping("/process-password-change")
    public ResponseEntity<UserInfoResponse> processPasswordChange(@RequestBody ChangePasswordFromLinkRequest request) {
        try {
            UserInfoResponse response = authenticationService.changePasswordLink(request);
            if (response.getResponse().equals("User with given e-mail does not exist.")) {
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
            } else if (!response.getResponse().equals("Password changed successfully.")) {
                return ResponseEntity.status(HttpStatus.CONFLICT).body(response);
            }
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(UserInfoResponse.builder().response("Other error: " + e.getMessage()).build());
        }
    }

    /**
     * Endpoint for deleting a user account.
     *
     * @param request The request containing the user's password.
     * @return ResponseEntity containing the user deletion response.
     */
    @DeleteMapping("/delete-user")
    public ResponseEntity<UserInfoResponse> deleteUser(@RequestBody UserDeleteRequest request) {
        try {
            UserInfoResponse response = authenticationService.deleteUser(request.getPassword());
            if (response.getResponse().equals("Incorrect password. User not deleted.")) {
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
            }
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(UserInfoResponse.builder().response("Other error: " + e.getMessage()).build());
        } catch (NoSuchElementException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(UserInfoResponse.builder().response("NoSuchElementException: " + "Invalid token").build());
        } catch (UsernameNotFoundException e ) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(UserInfoResponse.builder().response("User not found: " + e.getMessage()).build());
        }
    }

    /**
     * Endpoint for checking token validity and granting access.
     *
     * @param tokenValidityRequest The request containing the token to be checked.
     * @return ResponseEntity containing the token validity response.
     */
    @PostMapping("/access")
    public ResponseEntity<?> access(@RequestBody TokenValidityRequest tokenValidityRequest) {
        try {
            TokenValidityResponse response = authenticationService.checkAccess(tokenValidityRequest);
            if (!response.isAccess()) {
                return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(response);
            }
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(UserInfoResponse.builder().response("Other error: " + e.getMessage()).build());
        }
    }
}
