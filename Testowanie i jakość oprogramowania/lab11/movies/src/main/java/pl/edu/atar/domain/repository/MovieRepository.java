package pl.edu.atar.domain.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;
import pl.edu.atar.domain.entity.Movie;

import java.util.List;

@Repository
public interface MovieRepository extends JpaRepository<Movie, Long>, CrudRepository<Movie, Long> {

    @Query("SELECT m FROM Movie m ORDER BY m.releaseYear")
    List<Movie> findAll();
}
