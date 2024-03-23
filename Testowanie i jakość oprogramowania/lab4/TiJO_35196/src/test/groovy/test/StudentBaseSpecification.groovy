package test

import code.StudentBase
import spock.lang.Specification

class StudentBaseSpecification extends Specification {

    def "addStudent method adds a student to the list"() {
        given:
        def studentBase = new StudentBase()

        when:
        def result = studentBase.addStudent("Jakub", "Kowalski", "1")

        then:
            result == true
        and:
            StudentBase.Students.size() == 1
        and:
            StudentBase.Students[0].name == "Jakub"
        and:
            StudentBase.Students[0].surname == "Kowalski"
        and:
            StudentBase.Students[0].studentId == "1"
    }

    def "addStudent method data is null"() {
        given:
        def studentBase = new StudentBase()

        when:
        def result = studentBase.addStudent(null, "Kowalski", "1")

        then:
        result == false
    }

    def "updateStudent method data is null"(){
        given:
        def studentBase = new StudentBase()

        when:
        def result = studentBase.updateStudent("Adam", null, "1")

        then:
        result == false
    }

    def "removeStudent method data is null"(){
        given:
        def studentBase = new StudentBase()

        when:
        def result = studentBase.updateStudent(null, "Kowalski", "1")

        then:
        result == false
    }
}
