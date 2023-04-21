package org.example;

public class TabInfo {
    private String selectTab[] = {"imie, nazwisko, pesel, data_urodzenia", "nr_albumu, id_przedmiotu, termin_zaliczenia, ocena", "ulica, nr_domu, miejscowosc, kod_pocztowy", "imie, nazwisko, tytul, placa_zasadnicza", "id_fakultetu, nazwa_fakultetu, dostepny, id_prowadzacego", "nazwa_kierunku, tryb_studiow,tytul_zawodowy, stopien_studiow"};
    private String schematTab[] = {"dziekanat", "dziekanat", "dziekanat", "kadry", "dziekanat", "dziekanat"};
    private String tabelaTab[] = {"studenci", "oceny", "adresy", "prowadzacy", "fakultety", "kierunki_studiow"};
    private String studentNames[] = {"imie", "nazwisko", "pesel", "data_urodzenia"};
    private String gradesNames[] = {"nr_albumu", "id_przedmiotu", "termin_zaliczenia", "ocena"};
    private String addressNames[] = {"ulica", "nr_domu", "miejscowosc", "kod_pocztowy"};
    private String teachersNames[] = {"imie", "nazwisko", "tytul", "placa_zasadnicza"};
    private String FacultyNames[] = {"id_fakultetu", "nazwa_fakultetu", "dostepny", "id_prowadzacego"};
    private String DirectionNames[] = {"nazwa_kierunku", "tryb_studiow", "tytul_zawodowy", "stopien_studiow"};
    public String[] getSelectTab() { return selectTab; }
    public String[] getSchematTab() { return schematTab; }
    public String[] getTabelaTab() { return tabelaTab; }
    public String[] getStudentNames() { return studentNames; }
    public String[] getGradesNames() { return gradesNames; }
    public String[] getAddressNames() { return addressNames; }
    public String[] getTeachersNames() { return teachersNames; }
    public String[] getFacultyNames() { return FacultyNames; }
    public String[] getDirectionNames() { return DirectionNames; }

}
