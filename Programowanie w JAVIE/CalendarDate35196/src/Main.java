import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * The main class of the program
 *
 * @author Kamil Dereń
 */
public class Main {

    /**
     * Program operation
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        nextAndBackWeek();
        dateFromFile();
        formatData();
        dayWeekName();
        parseDate();
        ArrayListAndTab();
        mainManuEvents();
    }

    static void nextAndBackWeek(){
        System.out.println("\n1.    Add and Min week");
        CalendarDate cal1 = new CalendarDate();
        CalendarDate cal2 = new CalendarDate(29, 11, 2022);
        System.out.println(cal1);
        cal1.nextWeek();
        System.out.println("Add 7 days = " + cal1);
        System.out.println(cal2);
        cal2.backWeek();
        System.out.println("Min 7 days = " + cal2);
    }
    static void dateFromFile() throws IOException {
        System.out.println("\n2.    Date from file:");
        CalendarDate cal1 = new CalendarDate("data");
        System.out.println("Current Data:   "+cal1.showDate(1));
    }
    static void dayWeekName() {
        System.out.println("\n3.    Day week name:");
        CalendarDate cal1 = new CalendarDate(9, 1, 2023);
        System.out.println(cal1.showDate(2));
        System.out.println("The day of the week using References:  " + cal1.getWeekDayUsingReference());
        System.out.println("The day of the week using Modulo:      " + cal1.getWeekDayUsingModulo());
    }
    static void formatData() {
        System.out.println("\n4.    Displaying using the correct format:");
        CalendarDate cal1 = new CalendarDate(28, 3, 2024);
        System.out.println(cal1.showDate(1));
        System.out.println(cal1.showDate(2));
        System.out.println(cal1.showDate(3));
        System.out.println(cal1.showDate(4));
        System.out.println(cal1.showDate(0));
    }
    static void parseDate() {
        System.out.println("\n5.    Date create by parse: ");
        CalendarDate cal1 = CalendarDate.parse("15-1-1731");
        CalendarDate cal2 = CalendarDate.parse("9.3.2020r");
        System.out.println(cal1.showDate(1));
        System.out.println(cal2.showDate(2));
    }
    static void ArrayListAndTab(){
        System.out.println("\n6.    ArrayList and Tab:");
        CalendarDate cal1 = new CalendarDate(9, 1, 2001);
        CalendarDate cal2 = new CalendarDate(13, 4, 1984);
        CalendarDate cal3 = new CalendarDate(15, 3, 3000);
        ArrayList<CalendarDate> list = new ArrayList<CalendarDate>();
        list.add(cal1); list.add(cal2); list.add(cal3);
        System.out.println("List before sort");
        for(int n=0 ; n< list.size(); n++) System.out.println(list.get(n));
        Collections.sort(list, new SortDate());
        System.out.println("\nList after sort");
        for(int n=0 ; n< list.size(); n++) System.out.println(list.get(n));

        CalendarDate[] tab = {cal1,cal2,cal3};
        System.out.println("\nTab before sort");
        for(int n=0 ; n< tab.length; n++) System.out.println(tab[n]);
        Arrays.sort(tab, new SortDate());
        System.out.println("\nTab after sort");
        for(int n=0 ; n< tab.length; n++) System.out.println(tab[n]);
    }
    static void mainManuEvents() throws IOException {
        System.out.println("\n7.    Events Log:");
        CalendarDate cal1 = new CalendarDate(1,2,2000);
        cal1.mainMenu();
    }
}
