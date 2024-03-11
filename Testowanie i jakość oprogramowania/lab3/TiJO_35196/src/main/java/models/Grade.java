package models;

public class Grade {
    private String studentId;
    private String subject;
    private double grade;

    public Grade(String studentId, String subject, double grade) {
        this.studentId = studentId;
        this.subject = subject;
        this.grade = grade;
    }

    public String getStudentId() {
        return studentId;
    }

    public void setStudentId(String studentId) {
        this.studentId = studentId;
    }

    public String getSubject() {
        return subject;
    }

    public void setSubject(String subject) {
        this.subject = subject;
    }

    public double getGrade() {
        return grade;
    }

    public void setGrade(double grade) {
        this.grade = grade;
    }
}
