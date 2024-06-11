package pl.edu.atar.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import pl.edu.atar.domain.dto.RegisterUserDto;
import pl.edu.atar.domain.dto.ResponseData;
import pl.edu.atar.service.UserService;

@RestController
@RequestMapping(value="/api")
public class UserApiController {
    private static final Logger LOGGER = LoggerFactory.getLogger(UserApiController.class);

    private final UserService userService;

    public UserApiController(UserService userService) {
        this.userService = userService;
    }

    @PostMapping(value = "/users/register")
    public ResponseEntity<String> registerUser(@RequestBody RegisterUserDto registerUserDto) {
        LOGGER.info("register user: {}", registerUserDto);
        ResponseData responseData = userService.registerUser(registerUserDto);

        return new ResponseEntity<>(responseData.getInvalidFieldNames(), HttpStatus.valueOf(responseData.getErrorCode()));
    }

}
