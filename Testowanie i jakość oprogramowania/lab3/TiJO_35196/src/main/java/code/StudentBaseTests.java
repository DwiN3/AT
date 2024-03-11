package code;

import java.util.Random;

class StudentBaseTests {
    private static StudentBase studentBase = new StudentBase();
    private static StudentBaseTests studentBaseTests = new StudentBaseTests();

    public static void main(String[] args) {
        studentBaseTests.testAddStudentNameIsNull();
        studentBaseTests.testCorrectAddStudent();
        System.out.println("----------Add-Student----------");
        studentBase.showStudents();
        System.out.println("------------Update-Student----------");
        studentBaseTests.testUpdateStudentSurnameIsNull();
        studentBaseTests.testCorrectUpdateStudent();
        studentBase.showStudents();
        System.out.println("----------Remove Student----------");
        studentBaseTests.testRemoveStudentIndexIsNull();
        studentBaseTests.testCorrectRemoveStudent();
        studentBase.showStudents();
        System.out.println("----------Add Grade----------");
        studentBaseTests.testAddGradeGradeIsMinus();
        studentBaseTests.testCorrectAddGradeGrade();
        studentBase.showGrade();
        System.out.println("-------Calculate Average Grade------");
        studentBaseTests.testCalculateAverageGradeSubjectNameIsEmpty();
        studentBaseTests.testCorrectCalculateAverageGrade();
    }

    private void testAddStudentNameIsNull(){
        //Arrange
        String name = null;
        String surname = "Kowalski";
        String studentId = "2";

        // Act
        boolean result = studentBase.addStudent(name, surname, studentId);

        // Assert
        assert result == false : "Name is Null";
    }

    private void testCorrectAddStudent(){
        String name = "Adam";
        String surname = "Kowalski";
        String studentId = "1";
        studentBase.addStudent(name, surname, studentId);
    }

    private void testUpdateStudentSurnameIsNull(){
        //Arrange
        String name = "Adam";
        String surname = null;
        String studentId = "2";

        // Act
        boolean result = studentBase.updateStudent(name, surname, studentId);

        // Assert
        assert result == false : "Surname is Null";
    }

    private void testCorrectUpdateStudent(){
        String name = "Maciek";
        String surname = "Kowalski";
        String studentId = "1";
        studentBase.updateStudent(name, surname, studentId);
    }

    private void testRemoveStudentIndexIsNull(){
        //Arrange
        String studentId = null;

        // Act
        boolean result = studentBase.removeStudent(studentId);

        // Assert
        assert result == false : "Index is Null";
    }

    private void testCorrectRemoveStudent(){
        String studentId = "1";
        studentBase.removeStudent(studentId);
    }

    private void testAddGradeGradeIsMinus(){
        //Arrange
        String studentId = "1";
        String subject = "Przyroda";
        double grade = -5;

        // Act
        boolean result = studentBase.addGrade(studentId, subject, grade);

        // Assert
        assert result == false : "Grade is Minus";
    }

    private void testCorrectAddGradeGrade(){
        Random random = new Random();

        for(int i = 0; i < 4; i++) {
            String studentId = String.valueOf(random.nextInt(10) + 1);
            String subject = "Przyroda";
            double grade = (double) Math.round(2 + random.nextDouble() * (5 - 2));
            studentBase.addGrade(studentId, subject, grade);
        }
    }

    private void testCalculateAverageGradeSubjectNameIsEmpty(){
        //Arrange
        String subject = "";

        // Act
        double result = studentBase.calculateAverageGrade(subject);

        // Assert
        assert result < 0 : "No selected subject";
    }

    private void testCorrectCalculateAverageGrade(){
        String subject = "Przyroda";
        studentBase.calculateAverageGrade(subject);
    }
}