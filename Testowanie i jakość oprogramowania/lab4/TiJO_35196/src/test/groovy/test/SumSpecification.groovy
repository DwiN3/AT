package test

import spock.lang.Specification

class SumSpecification extends Specification {

    def "one plus two should equal three"(){

        given:
            def one = 1
            def two = 2

        when:
            def result = one + two

        then:
            result == 3
    }
}
