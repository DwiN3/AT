package org.example;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;

public class TabSet {
    private String select;
    private String schemat;
    private String tabela;
    private String from;
    private String[] List1 = null, List2 = null, List3 = null, List4 = null;
    private String selectNames[] = new String[4];
    private Connection connect;

    public TabSet(Connection connect, int opction )  {
        TabInfo tab = new TabInfo();

        String[] selectTab = tab.getSelectTab();
        String[] schematTab = tab.getSchematTab();
        String[] tabelaTab = tab.getTabelaTab();

        select = selectTab[opction];
        schemat = schematTab[opction];
        tabela = tabelaTab[opction];
        from = schemat+"."+ tabela;

        if(opction == 0) selectNames = tab.getStudentNames();
        if(opction == 1) selectNames = tab.getGradesNames();
        if(opction == 2) selectNames = tab.getAddressNames();
        if(opction == 3) selectNames = tab.getTeachersNames();
        if(opction == 4) selectNames = tab.getFacultyNames();
        if(opction == 5) selectNames = tab.getDirectionNames();

        this.connect = connect;
    }

    public void runSetTabel() {
        try {
            Statement stmt = connect.createStatement();
            ResultSet rs = stmt.executeQuery("SELECT " + select+" FROM "+from);
            ArrayList<String> list1 = new ArrayList<String>();
            ArrayList<String> list2 = new ArrayList<String>();
            ArrayList<String> list3 = new ArrayList<String>();
            ArrayList<String> lista4 = new ArrayList<String>();

            while(rs.next()) {
                list1.add(rs.getString(selectNames[0]));
                list2.add(rs.getString(selectNames[1]));
                list3.add(rs.getString(selectNames[2]));
                lista4.add(rs.getString(selectNames[3]));
            }
            List1 = list1.toArray(new String[0]);
            List2 = list2.toArray(new String[0]);
            List3 = list3.toArray(new String[0]);
            List4 = lista4.toArray(new String[0]);

        } catch(SQLException e) {
            System.out.println(e.getMessage());
        }
        //for(int n=0; n < selectedName.length ;n++) System.out.println(selectedName[n]+ "    ");;
    }

    public String[] getList1() { return List1; }
    public String[] getList2() { return List2; }
    public String[] getList3() { return List3;}
    public String[] getList4() { return List4;}
}
