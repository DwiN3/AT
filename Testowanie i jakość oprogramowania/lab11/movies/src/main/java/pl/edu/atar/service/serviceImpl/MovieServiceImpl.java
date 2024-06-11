package pl.edu.atar.service.serviceImpl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import pl.edu.atar.domain.dto.*;
import pl.edu.atar.domain.entity.Movie;
import pl.edu.atar.domain.mapper.MovieListMapper;
import pl.edu.atar.domain.repository.MovieRepository;
import pl.edu.atar.service.MovieService;

import java.util.List;

@Service
public class MovieServiceImpl implements MovieService {

    private static final Logger LOGGER = LoggerFactory.getLogger(MovieServiceImpl.class);

    private final MovieRepository movieRepository;
    private final MovieListMapper movieListMapper;

    @Autowired
    public MovieServiceImpl(MovieRepository movieRepository,
                            MovieListMapper movieListMapper) {

        this.movieRepository = movieRepository;
        this.movieListMapper = movieListMapper;
    }

    @Override
    public List<MovieDto> findAll() {
        List<Movie> movies = movieRepository.findAll();
        return movieListMapper.mapToDto(movies);
    }
}
