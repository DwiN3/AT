package org.example;

import connection.Connect;

import java.io.IOException;
import java.sql.Connection;

public class Management {
    private String[] column1, column2, column3, column4;
    private String[] text = new String[4];
    private String[] nameColumn = new String[4];

    public Management(int opction) throws IOException {
        Connect connect = new Connect();
        Connection con = connect.getConnection();

        TabSet tabSet = new TabSet(con, opction);
        tabSet.runSetTabel();

        if(opction == 0){
            text[0] ="name";
            text[1] ="surname";
            text[2] ="pesel";
            text[3] ="borndate";
            nameColumn[0] = "Imie";
            nameColumn[1] = "Nazwisko";
            nameColumn[2] = "Pesel";
            nameColumn[3] = "Data Urodzenia";
        }
        if(opction == 1){
            text[0] ="nrindex";
            text[1] ="subject";
            text[2] ="passdate";
            text[3] ="grade";
            nameColumn[0] = "Nr albumu";
            nameColumn[1] = "Przedmiot";
            nameColumn[2] = "Termin Zaliczenia";
            nameColumn[3] = "Ocena";
        }
        if(opction == 2){
            text[0] ="street";
            text[1] ="nrhouse";
            text[2] ="locality";
            text[3] ="zipcode";
            nameColumn[0] = "Ulica";
            nameColumn[1] = "Nr Domu";
            nameColumn[2] = "Miejscowosc";
            nameColumn[3] = "Nr pocztowy";
        }
        if(opction == 3){
            text[0] ="nametprof";
            text[1] ="surnameprof";
            text[2] ="title";
            text[3] ="wage";
            nameColumn[0] = "Imie";
            nameColumn[1] = "Nazwisko";
            nameColumn[2] = "Tytul";
            nameColumn[3] = "Placa";
        }
        if(opction == 4){
            text[0] ="idf";
            text[1] ="namef";
            text[2] ="access";
            text[3] ="teacherf";
            nameColumn[0] = "ID Fakultetu";
            nameColumn[1] = "Nazwa";
            nameColumn[2] = "Dostep";
            nameColumn[3] = "ID Prowadzacego";
        }
        if(opction == 5){
            text[0] ="namek";
            text[1] ="type";
            text[2] ="titlek";
            text[3] ="academicdegree";
            nameColumn[0] = "Nazwa Kierunku";
            nameColumn[1] = "Typ studiow";
            nameColumn[2] = "Tytul zawodowy";
            nameColumn[3] = "Stopien naukowy";
        }
        column1 = tabSet.getList1();
        column2 = tabSet.getList2();
        column3 = tabSet.getList3();
        column4 = tabSet.getList4();
        connect.close();
    }

    public String[] getColumn1() { return column1; }
    public String[] getColumn2() { return column2; }
    public String[] getColumn3() { return column3; }
    public String[] getColumn4() { return column4; }
    public String getText1() { return text[0]; }
    public String getText2() { return text[1]; }
    public String getText3() { return text[2]; }
    public String getText4() { return text[3]; }
    public String getNameColumn1() { return nameColumn[0];}
    public String getNameColumn2() { return nameColumn[1];}
    public String getNameColumn3() { return nameColumn[2];}
    public String getNameColumn4() { return nameColumn[3];}
}