package pl.edu.atar.domain.validators;

import pl.edu.atar.domain.forms.RegisterFormFields;

public class LoginValidator implements Validator {

    private final String login;

    public LoginValidator(final String login) {
        this.login = login;
    }

    public boolean isValid() {
        final int LOGIN_MIN_LENGTH = 4;

        if(login == null || login.length() < LOGIN_MIN_LENGTH) {
            return false;
        }

        return true;
    }

    public String fieldName() {
        return RegisterFormFields.LOGIN.fieldName();
    }
}