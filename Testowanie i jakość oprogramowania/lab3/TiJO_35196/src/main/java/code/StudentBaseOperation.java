package code;

public interface StudentBaseOperation {
    void showStudents();
    void showGrade();
    boolean addStudent(String name, String surname, String studentId);
    boolean updateStudent(String name, String surname, String studentId);
    boolean removeStudent(String studentID);
    boolean addGrade(String studentID, String subject, double grade);
    double calculateAverageGrade(String subject);
}
