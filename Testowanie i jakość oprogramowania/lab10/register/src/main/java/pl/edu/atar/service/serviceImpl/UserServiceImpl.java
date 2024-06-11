package pl.edu.atar.service.serviceImpl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import pl.edu.atar.domain.dto.RegisterUserDto;
import pl.edu.atar.domain.dto.ResponseData;
import pl.edu.atar.domain.validators.*;
import pl.edu.atar.service.UserService;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class UserServiceImpl implements UserService {

    private static final Logger LOGGER = LoggerFactory.getLogger(UserServiceImpl.class);

    @Override
    public ResponseData registerUser(RegisterUserDto registerUserDto) {

        List<Validator> validators = List.of(
                new LoginValidator(registerUserDto.getLogin()),
                new FirstNameValidator(registerUserDto.getFirstName()),
                new LastNameValidator(registerUserDto.getLastName()),
                new PasswordValidator(registerUserDto.getPassword()),
                new PeselValidator(registerUserDto.getPesel())
        );

        List<String> invalidFieldNames = validators.stream()
                .filter(field -> !field.isValid())
                .map(Validator::fieldName)
                .collect(Collectors.toList());

        ResponseData responseData = new ResponseData();
        responseData.addInvalidFieldNames(invalidFieldNames);

        return responseData;
    }
}