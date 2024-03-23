package code

import exceptions.InsufficientFundsException
import exceptions.InvalidPinException
import spock.lang.Specification

class ATMSpecification extends Specification{
    def "pin is not correct"(){
        given: "given pin 1233 correct pin 1234"
            def atm = new ATM()

        when: "give wrong pin"
            def exception = null
            try {
                atm.checkBalance(1233)
            } catch (InvalidPinException e) {
                exception = e
            }

        then: "An InvalidPinException is thrown"
        exception != null
        exception.message == "Błędny pin"
    }

    def "deposit with uncorrect pin"(){
        given: "given pin 1233 correct pin 1234"
        def atm = new ATM()

        when: "give wrong pin"
        def exception = null
        try {
            atm.deposit(1233,20)
        } catch (InvalidPinException e) {
            exception = e
        }

        then: "An InvalidPinException is thrown"
        exception != null
        exception.message == "Błędny pin"
    }

    def "withdraw with uncorrect pin"(){
        given: "given pin 1233 correct pin 1234"
        def atm = new ATM()

        when: "give wrong pin"
        def exception = null
        try {
            atm.withdraw(1233,20)
        } catch (InvalidPinException e) {
            exception = e
        }

        then: "An InvalidPinException is thrown"
        exception != null
        exception.message == "Błędny pin"
    }

    def "withdraw with not enough budget"(){
        given: "wanted to withdraw 501 but budget is 500"
        def atm = new ATM()

        when: "give not enough value of money to withdraw"
        def exception = null
        try {
            atm.withdraw(1234,501)
        } catch (InsufficientFundsException e) {
            exception = e
        }

        then: "An InsufficientFundsException is thrown"
        exception != null
        exception.message == "Brak wymagających środków"
    }
}
