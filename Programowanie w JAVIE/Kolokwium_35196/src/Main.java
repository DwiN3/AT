import java.util.ArrayList;
import java.util.Collections;

public class Main {
    public static void main(String[] args) throws EmptyInfo {
        Student student1 = new Student("Michal","Kolwalski",3512);
        Przedmiot przedmiot1 = new Przedmiot("laboratorium","Algorytmy",4);
        Przedmiot przedmiot2 = new Przedmiot("wyklad","Baza danych",5);

        Student student2 = new Student("Andrzej","Nowak",2412);
        Przedmiot przedmiot3 = new Przedmiot("laboratorium","Programowanie Java",2);
        Przedmiot przedmiot4 = new Przedmiot("wyklad","Fizyka",5);

        Student student3 = new Student("Daniel","Kac",4000);
        Przedmiot przedmiot5 = new Przedmiot("laboratorium","Sieci komputerowe",5);
        Przedmiot przedmiot6 = new Przedmiot("wyklad","Systemy operacyjne",3);

        student1.dodajPrzedmit(przedmiot1);
        student1.dodajPrzedmit(przedmiot2);
        student1.obliczSrednia();
        System.out.println("\nImie studenta 1: "+student1.getImie());
        System.out.println("Srednia studenta 1: "+student1.getSrednia());

        student2.dodajPrzedmit(przedmiot3);
        student2.dodajPrzedmit(przedmiot4);
        student2.obliczSrednia();
        System.out.println("\nImie studenta 2: "+student2.getImie());
        System.out.println("Srednia studenta 2: "+student2.getSrednia());

        student3.dodajPrzedmit(przedmiot5);
        student3.dodajPrzedmit(przedmiot6);
        student3.obliczSrednia();
        System.out.println("\nImie studenta 3: "+student3.getImie());
        System.out.println("Srednia studenta 3: "+student3.getSrednia());
        student3.usunPrzedmiot("Sieci komputerowe");
        student3.obliczSrednia();
        System.out.println("Srednia studenta 3 po usunieciu przedmiotu: "+student3.getSrednia());

        ArrayList<Student> listaStudentow = new ArrayList<Student>();
        listaStudentow.add(student3); listaStudentow.add(student2); listaStudentow.add(student1);

        System.out.println("\nLista przed sort:");
        for(int n=0 ; n< listaStudentow.size(); n++) System.out.println("Nr: "+listaStudentow.get(n).getNumerAlbumu()+"    Imie: "+(listaStudentow.get(n)).getImie());
        Collections.sort(listaStudentow);
        System.out.println("\nLista po sort:");
        for(int n=0 ; n< listaStudentow.size(); n++) System.out.println("Nr: "+listaStudentow.get(n).getNumerAlbumu()+"    Imie: "+(listaStudentow.get(n)).getImie());
        System.out.println("\n");

        //Przedmiot przedmiotERROR = new Przedmiot("","ErrorTest",2);
    }
}