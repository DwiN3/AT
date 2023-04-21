package Bases;

public class KierunkiStudiow {
    private String namek;
    private String type;
    private String titlek;
    private String academicdegree;

    public KierunkiStudiow(String namek, String type, String titlek, String academicdegree) {
        this.namek = namek;
        this.type = type;
        this.titlek = titlek;
        this.academicdegree = academicdegree;
    }

    public String getNamek() { return namek; }

    public String getType() { return type; }

    public String getTitlek() { return titlek; }

    public String getAcademicdegree() { return academicdegree; }
}
