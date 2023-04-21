package Bases;

public class Student {
    private String name;
    private String surname;
    private String pesel;
    private String borndate;

    public Student(String name, String surname, String pesel, String borndate) {
        this.name = name;
        this.surname = surname;
        this.pesel = pesel;
        this.borndate = borndate;
    }

    public String getSurname() { return surname;}
    public String getName() { return name; }
    public String getPesel() { return pesel; }
    public String getBorndate() { return borndate; }
}