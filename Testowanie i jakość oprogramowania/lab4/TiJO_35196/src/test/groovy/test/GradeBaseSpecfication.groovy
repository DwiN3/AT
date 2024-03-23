package test

import code.StudentBase
import spock.lang.Specification

class GradeBaseSpecfication extends Specification {
    def "addGrade method adds a minus grade to the list"() {
        given:
        def studentBase = new StudentBase()

        when:
        def result = studentBase.addGrade("1", "Przyroda",-5)

        then:
            result == false
    }

    def "calculate average grade subject name is empty"() {
        given:
        def studentBase = new StudentBase()

        when:
        def result = studentBase.calculateAverageGrade("")

        then:
        result == -1
    }
}
