package code;

import exceptions.InsufficientFundsException;
import exceptions.InvalidPinException;

public interface ATMInterface {

    /**
     * Metoda do sprawdzania salda na koncie.
     * @param pin PIN użytkownika.
     * @return Saldo konta użytkownika.
     * @throws InvalidPinException Jeśli PIN jest nieprawidłowy.
     */
    double checkBalance(int pin) throws InvalidPinException;

    /**
     * Metoda do wpłacania środków na konto.
     * @param pin PIN użytkownika.
     * @param amount Kwota do wpłacenia.
     * @return Aktualne saldo konta po dokonaniu wpłaty.
     * @throws InvalidPinException Jeśli PIN jest nieprawidłowy.
     */
    double deposit(int pin, double amount) throws InvalidPinException;

    /**
    * Metoda do wypłacania środków z konta.
    * @param pin PIN użytkownika.
    * @param amount Kwota do wypłacenia.
    * @return Aktualne saldo konta po dokonaniu wypłaty.
    * @throws InsufficientFundsException Jeśli saldo jest niewystarczające.
    * @throws InvalidPinException Jeśli PIN jest nieprawidłowy.
    */
    double withdraw(int pin, double amount) throws InsufficientFundsException, InvalidPinException;
}
