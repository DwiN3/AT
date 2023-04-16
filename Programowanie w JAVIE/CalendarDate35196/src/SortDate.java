import java.util.Comparator;

/**
 * The class that sorts stored dates in order from smallest to largest
 *
 * @author Kamil Dereń
 */
public class SortDate implements Comparator<CalendarDate> {
    /**
     * A function that compares two dates
     * @param date1 the first object to be compared.
     * @param date2 the second object to be compared.
     * @return Returns the correct date order
     */
        public int compare(CalendarDate date1, CalendarDate date2) {
            int year1 = date1.getYear();
            int year2 = date2.getYear();
            int month1 = date1.getMonth();
            int month2 = date2.getMonth();
            int day1 = date1.getDay();
            int day2 = date2.getDay();
            if (year1 != year2) return year1 - year2;
            else if (month1 != month2) return month1 - month2;
            else return day1 - day2;
        }
}
