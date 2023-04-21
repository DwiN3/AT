public class Przedmiot {
    private String nazwa;
    private String typPrzedmiotu;
    private int ocena = 0;

    public Przedmiot(String typPrzedmiotu_, String nazwa_, int ocena_) throws EmptyInfo {
        if (typPrzedmiotu_ == null || typPrzedmiotu_.isEmpty() || nazwa_ == null || nazwa_.isEmpty()) {
            throw new EmptyInfo("Pusty parametr");
        }
        this.nazwa = nazwa_;
        this.typPrzedmiotu = typPrzedmiotu_;
        if(ocena_ > 1 && ocena_ < 6) this.ocena = ocena_;
        else this.ocena = 2;
    }

    public String getNazwa() { return nazwa; }

    public String getTypPrzedmiotu() { return typPrzedmiotu; }
    public int getOcena() { return ocena; }
}