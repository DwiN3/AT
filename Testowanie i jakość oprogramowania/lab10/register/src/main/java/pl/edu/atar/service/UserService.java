package pl.edu.atar.service;

import pl.edu.atar.domain.dto.*;

public interface UserService {

    ResponseData registerUser(RegisterUserDto createMovieDto);
}
