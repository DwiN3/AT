package pl.edu.atar.domain.validators;

public interface Validator {

    boolean isValid();

    String fieldName();
}
