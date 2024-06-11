package code

import spock.lang.Specification

class LibraryStubSpec extends Specification{
    def "Test amount of book"() {
        given:
            def bookManagementStub = Stub(BookManagment)
            def calcForStub = new Library(bookManagementStub)

        when:
            calcForStub.returnBook("Title 1", "Author 1", 1)
            calcForStub.returnBook("Title 2", "Author 2", 2)

        then:
            calcForStub.books().size() == 2
    }
}
