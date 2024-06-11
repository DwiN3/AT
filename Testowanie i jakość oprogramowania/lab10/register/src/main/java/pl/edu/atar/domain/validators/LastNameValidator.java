package pl.edu.atar.domain.validators;

import pl.edu.atar.domain.forms.RegisterFormFields;

public class LastNameValidator implements Validator {

    private final String lastName;

    public LastNameValidator(final String lastName) {
        this.lastName = lastName;
    }

    public boolean isValid() {

        if(lastName == null || lastName.isBlank()) {
            return false;
        }

        return true;
    }

    public String fieldName() {
        return RegisterFormFields.LAST_NAME.fieldName();
    }
}