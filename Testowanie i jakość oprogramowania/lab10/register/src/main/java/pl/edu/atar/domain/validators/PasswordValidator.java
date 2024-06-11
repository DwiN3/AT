package pl.edu.atar.domain.validators;

import pl.edu.atar.domain.forms.RegisterFormFields;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PasswordValidator implements Validator {

    private final String password;

    public PasswordValidator(final String password) {
        this.password = password;
    }

    public boolean isValid() {
        final int PASSWORD_MIN_LENGTH = 4;

        if(password== null || password.length() < PASSWORD_MIN_LENGTH) {
            return false;
        }

        String regex = "^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()_+=-]).{4,}$";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(password);

        return matcher.matches();
    }

    public String fieldName() {
        return RegisterFormFields.PASSWORD.fieldName();
    }
}