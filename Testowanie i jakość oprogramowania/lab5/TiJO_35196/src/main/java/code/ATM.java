package code;

import exceptions.InsufficientFundsException;
import exceptions.InvalidPinException;

public class ATM implements ATMInterface {
    private int correctPin = 1234;
    private double amountMoneyOnAccount = 500;
    @Override
    public double checkBalance(int pin) throws InvalidPinException {
        if(pin == correctPin) return 1000;
        else throw new InvalidPinException("Błędny pin");
    }

    @Override
    public double deposit(int pin, double amount) throws InvalidPinException {
        if(pin == correctPin) return amountMoneyOnAccount + amount;
        else throw new InvalidPinException("Błędny pin");
    }

    @Override
    public double withdraw(int pin, double amount) throws InsufficientFundsException, InvalidPinException {
        if (pin != correctPin) {
            throw new InvalidPinException("Błędny pin");
        }

        if (amount > amountMoneyOnAccount) {
            throw new InsufficientFundsException("Brak wymagających środków");
        }

        amountMoneyOnAccount -= amount;
        return amount;
    }
}
