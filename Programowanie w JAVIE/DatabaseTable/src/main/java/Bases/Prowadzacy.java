package Bases;

public class Prowadzacy {
    private String nametprof;
    private String surnameprof;
    private String title;
    private String wage;

    public Prowadzacy(String nametprof, String surnameprof, String title, String wage) {
        this.nametprof = nametprof;
        this.surnameprof = surnameprof;
        this.title = title;
        this.wage = wage;
    }

    public String getNametprof() {
        return nametprof;
    }

    public String getSurnameprof() {
        return surnameprof;
    }

    public String getTitle() {
        return title;
    }

    public String getWage() {
        return wage;
    }
}
