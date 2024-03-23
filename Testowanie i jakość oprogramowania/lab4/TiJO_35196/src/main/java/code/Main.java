package code;

import java.util.Random;

class Main {
    static StudentBase studentBase = new StudentBase();

    public static void main(String[] args) {
        System.out.println("----------Add-Student----------");
        studentBase.addStudent("Adam", "Kowalski", "1");
        studentBase.showStudents();

        System.out.println("------------Update-Student----------");
        studentBase.updateStudent("Maciek", "Kowalski", "1");
        studentBase.showStudents();

        System.out.println("----------Remove Student----------");
        studentBase.removeStudent("1");
        studentBase.showStudents();

        System.out.println("----------Add Grade----------");
        studentBase.removeStudent("1");

        studentBase.addGrade("2", "Przyroda", 4);
        studentBase.showGrade();
        
        System.out.println("-------Calculate Average Grade------");
        Random random = new Random();
        for(int i = 0; i < 4; i++) {
            String studentId = String.valueOf(random.nextInt(10) + 1);
            String subject = "Przyroda";
            double grade = (double) Math.round(2 + random.nextDouble() * (5 - 2));
            studentBase.addGrade(studentId, subject, grade);
        }
        studentBase.calculateAverageGrade("Przyroda");
    }
}