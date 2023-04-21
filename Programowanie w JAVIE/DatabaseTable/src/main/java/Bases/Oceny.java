package Bases;

public class Oceny {
    private String nrindex;
    private String subject;
    private String passdate;
    private String grade;

    public Oceny(String nrindex, String subject, String passdate, String grade) {
        this.nrindex = nrindex;
        this.subject = subject;
        this.passdate = passdate;
        this.grade = grade;
    }

    public String getNrindex() { return nrindex; }
    public String getSubject() { return subject; }
    public String getPassdate() { return passdate; }
    public String getGrade() { return grade; }
}
