package pl.edu.atar.domain.dto;

import java.io.Serializable;

public class MovieDto implements Serializable {

    private Long movieId;
    private String title;
    private String image;
    private int releaseYear;

    public MovieDto(Builder builder){
        movieId = builder.movieId;
        title = builder.title;
        image = builder.image;
        releaseYear = builder.releaseYear;
    }

    public MovieDto() {
    }

    public Long getMovieId() {
        return movieId;
    }

    public String getTitle() {
        return title;
    }

    public String getImage() {
        return image;
    }

    public int getReleaseYear() {
        return releaseYear;
    }

    public void setMovieId(Long movieId) {
        this.movieId = movieId;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public void setImage(String image) {
        this.image = image;
    }

    public void setReleaseYear(int releaseYear) {
        this.releaseYear = releaseYear;
    }

    public static class Builder {
        private Long movieId;
        private String title;
        private String image;
        private int releaseYear;

        public Builder(){ }

        public Builder movieId(Long movieId) {
            this.movieId = movieId;
            return this;
        }

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        public Builder image(String image) {
            this.image = image;
            return this;
        }

        public Builder releaseYear(int releaseYear) {
            this.releaseYear = releaseYear;
            return this;
        }

        public Builder fromPrototype(MovieDto prototype){
            this.movieId = prototype.getMovieId();
            this.title = prototype.getTitle();
            this.image = prototype.getImage();
            this.releaseYear = prototype.getReleaseYear();
            return this;
        }

        public MovieDto build() {
            return new MovieDto(this);
        }
    }
}
