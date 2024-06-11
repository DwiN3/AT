package pl.edu.atar.domain.dto;

import java.io.Serializable;

public class RegisterUserDto implements Serializable {
    private String login;
    private String firstName;
    private String lastName;
    private String password;
    private String pesel;

    public RegisterUserDto() {
    }

    public String getLogin() {
        return login;
    }

    public String getFirstName() {
        return firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public String getPassword() {
        return password;
    }

    public String getPesel() {
        return pesel;
    }

    @Override
    public String toString() {
        return "RegisterUserDto{" +
                "login='" + login + '\'' +
                ", firstName='" + firstName + '\'' +
                ", lastName='" + lastName + '\'' +
                ", password='" + password + '\'' +
                ", pesel='" + pesel + '\'' +
                '}';
    }
}
