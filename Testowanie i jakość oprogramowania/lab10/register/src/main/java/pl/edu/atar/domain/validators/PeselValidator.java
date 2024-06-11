package pl.edu.atar.domain.validators;

import pl.edu.atar.domain.forms.RegisterFormFields;

public class PeselValidator implements Validator {

    private final String pesel;

    public PeselValidator(final String pesel) {
        this.pesel = pesel;
    }

    public boolean isValid() {

        final int PESEL_LENGTH = 11;

        if(pesel == null || pesel.isEmpty() || pesel.length() != PESEL_LENGTH) {
            return false;
        }

        int[] wk = {1,3,7,9,1,3,7,9,1,3};
        int sum = 0;

        for(int i=0 ; i < PESEL_LENGTH-1 ; i++){
            sum += Character.getNumericValue(pesel.charAt(i)) * wk[i];
        }

        int controlDigit = Character.getNumericValue(pesel.charAt(PESEL_LENGTH - 1));

        int calculatedControlDigit = 10 - (sum % 10);
        if (calculatedControlDigit == 10) {
            calculatedControlDigit = 0;
        }

        return controlDigit == calculatedControlDigit;
    }

    public String fieldName() {
        return RegisterFormFields.PESEL.fieldName();
    }
}