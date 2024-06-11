package pl.edu.atar.domain.validators;

import pl.edu.atar.domain.forms.RegisterFormFields;

public class FirstNameValidator implements Validator {

    private final String firstName;

    public FirstNameValidator(final String firstName) {
        this.firstName = firstName;
    }

    public boolean isValid() {

        if(firstName == null || firstName.isBlank()) {
            return false;
        }

        return true;
    }

    public String fieldName() {
        return RegisterFormFields.FIRST_NAME.fieldName();
    }
}