package code;

import models.Grade;
import models.Student;

import java.util.ArrayList;
import java.util.List;

public class StudentBase implements StudentBaseOperation {
    public static List<Student> Students = new ArrayList<>();
    public static List<Grade> Grades = new ArrayList<>();

    @Override
    public void showStudents() {
        for(int i=0 ; i < Students.size() ; i++){
            System.out.println("Student nr: "+(i+1));
            System.out.println("Imie: "+Students.get(i).getName());
            System.out.println("Nazwisko: "+Students.get(i).getSurname());
            System.out.println("Id Ucznia: "+Students.get(i).getStudentId()+"\n");
        }
    }

    @Override
    public void showGrade() {
        for(int i=0 ; i < Grades.size() ; i++){
            System.out.println("Ocena nr: "+(i+1));
            System.out.println("Nazwa przedmiotu: "+Grades.get(i).getSubject());
            System.out.println("Ocena: "+Grades.get(i).getGrade());
            System.out.println("Id Ucznia: "+Grades.get(i).getStudentId()+"\n");
        }
    }

    @Override
    public boolean addStudent(String name, String surname, String studentId) {
        if(name == null || surname == null || studentId == null){
            return false;
        }

        Student newStudent = new Student(name,surname,studentId);
        Students.add(newStudent);
        return true;
    }
    @Override
    public boolean updateStudent(String name, String surname, String studentId) {
        if (name == null || surname == null || studentId == null) {
            return false;
        }

        for (int i = 0; i < Students.size(); i++) {
            if (studentId.equals(Students.get(i).getStudentId())) {
                Students.get(i).setName(name);
                Students.get(i).setSurname(surname);
            }
        }
        return true;
    }
    @Override
    public boolean removeStudent(String studentID){
        if (studentID == null) {
            return false;
        }

        for (int i = 0; i < Students.size(); i++) {
            if (studentID.equals(Students.get(i).getStudentId())) {
                Students.remove(i);
            }
            else {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean addGrade(String studentID, String subject, double grade) {
        if(grade < 1 || grade > 5 || studentID == null || subject == null){
           return false;
        }

        Grade newGrade = new Grade(studentID,subject,grade);
        Grades.add(newGrade);
        return true;
    }

    @Override
    public double calculateAverageGrade(String subject) {
        if (subject.isEmpty()) {
            return -1;
        }

        int count = 0;
        int sumGrades = 0;
        for (int i = 0; i < Grades.size(); i++) {
            if (subject.equals(Grades.get(i).getSubject())) {
                sumGrades += Grades.get(i).getGrade();
                count++;
            }
        }

        if (count == 0) {
            System.out.println("Brak danych dla przedmiotu: " + subject);
            return 0;
        } else {
            System.out.println("Średnia dla przedmiotu " + subject+": "+sumGrades/count);
            return sumGrades/count;
        }
    }
}
