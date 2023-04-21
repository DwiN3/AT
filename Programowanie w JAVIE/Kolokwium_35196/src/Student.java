import java.util.ArrayList;

public class Student implements Comparable<Student> {
    private String imie, nazwisko;
    int numerAlbumu;
    private double srednia;
    private ArrayList<Przedmiot> listaPrzedmiotow;

    public Student(String imie_,String nazwisko_, int numerAlbumu_){
        this.imie = imie_;
        this.nazwisko=nazwisko_;
        this.numerAlbumu=numerAlbumu_;
        listaPrzedmiotow = new ArrayList<Przedmiot>();
    }

    public void dodajPrzedmit(Przedmiot dodawanyPrzedmiot)
    {
        this.listaPrzedmiotow.add(dodawanyPrzedmiot);
    }

    public void usunPrzedmiot(String nazwaPrzedmiotu) {
        for (int i = 0; i < this.listaPrzedmiotow.size(); i++) {
            if (this.listaPrzedmiotow.get(i).getNazwa().equals(nazwaPrzedmiotu)) {
                this.listaPrzedmiotow.remove(i);
            }
        }
    }
    public void obliczSrednia(){
        if(!listaPrzedmiotow.isEmpty()) {
            this.srednia = 0;
            for (int i = 0; i < this.listaPrzedmiotow.size(); i++) {
                this.srednia += listaPrzedmiotow.get(i).getOcena();
            }
            this.srednia = this.srednia / listaPrzedmiotow.size();
        }
    }

    public String getImie() { return imie; }

    public String getNazwisko() { return nazwisko; }

    public int getNumerAlbumu() { return numerAlbumu; }

    public double getSrednia() { return srednia; }

    public ArrayList<Przedmiot> getListaPrzedmiotow() { return listaPrzedmiotow; }

    public int compareTo(Student student)
    {
        if(this.getNumerAlbumu() > student.getNumerAlbumu()) return 1;
        else if(this.getNumerAlbumu() < student.getNumerAlbumu()) return -1;
        else return 0;
    }
}