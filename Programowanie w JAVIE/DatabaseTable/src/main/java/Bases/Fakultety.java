package Bases;

public class Fakultety {
    private String idf;
    private String namef;
    private String access;
    private String teacherf;

    public Fakultety(String idf, String namef, String access, String teacherf) {
        this.idf = idf;
        this.namef = namef;
        this.access = access;
        this.teacherf = teacherf;
    }

    public String getIdf() {
        return idf;
    }

    public String getNamef() {
        return namef;
    }

    public String getAccess() {
        return access;
    }

    public String getTeacherf() {
        return teacherf;
    }
}
