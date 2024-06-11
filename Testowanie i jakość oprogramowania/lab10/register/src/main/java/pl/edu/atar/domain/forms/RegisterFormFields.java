package pl.edu.atar.domain.forms;

public enum RegisterFormFields {
    LOGIN("login"),
    FIRST_NAME("firstName"),
    LAST_NAME("lastName"),
    PASSWORD("password"),
    PESEL("pesel");

    private final String fieldName;

    RegisterFormFields(String fieldName) {
        this.fieldName = fieldName;
    }

    public String fieldName() {
        return this.fieldName;
    }
}
