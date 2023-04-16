import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.util.Calendar;
import java.util.Date;
import java.util.Scanner;

/**
 * The class for displaying a date with the ability to add or subtract a week
 *
 * @author Kamil Dereń
 */
public class CalendarDate {
    private int day, month, year; // Variables containing information about day, month and year
    private static int dayParse, monthParse, yearParse; // Variables containing information about the day, month and year, set in the convertDate function

    /**
     * The function stores the day
     * @return Returns the stored day
     */
    public int getDay() { return day; }

    /**
     * The function stores the month
     * @return Returns the stored month
     */
    public int getMonth() { return month; }

    /**
     * The function stores the year
     * @return Returns the stored year
     */
    public int getYear() { return year; }

    /**
     *  Retrieving the actual date
     */
    public CalendarDate(){
        LocalDate currentdate = LocalDate.now();
        this.day = currentdate.getDayOfMonth();
        this.month = currentdate.getMonthValue();
        this.year = currentdate.getYear();
    }

    /**
     * Retrieving date and setting according to entered values
     * @param _day Day typed
     * @param _month Month typed
     * @param _year Year typed
     */
    public CalendarDate(int _day, int _month, int _year) {
        this.day = _day;
        this.month = _month;
        this.year = _year;
    }

    /**
     * Retrieving date and setting according to entered values
     * @param nameFile Date file name
     */
    public CalendarDate(String nameFile) throws IOException {
        String data = "";
        try( BufferedReader br = new BufferedReader(new FileReader("./files/"+nameFile+".txt"))){
            String s;
            while ( (s = br.readLine()) != null ){
                data +=s;
            }
        }
        convertDate(data);
        this.day = dayParse;
        this.month = monthParse;
        this.year = yearParse;
    }

    /**
     * The function creates an object by entering the name text
     * @param data A variable that stores the entered date
     * @return The returned object that has the date entered
     */
    public static CalendarDate parse(String data) {
        convertDate(data);
        return new CalendarDate(dayParse, monthParse, yearParse);
    }

    /**
     * The function converts the text date so that the program can understand it
     * @param data Date stored
     */
    private static void convertDate(String data) {
        data = data.replace('.', '-');
        data = data.replace('/', '-');
        data = data.replace(' ', '-');
        data = data.replaceAll("[a-zA-Z]", "");
        try{
            SimpleDateFormat sdf = new SimpleDateFormat("dd-MM-yyyy");
            Date dataSDF = sdf.parse(data);
            Calendar calendar = Calendar.getInstance();
            calendar.setTime(dataSDF);
            dayParse = calendar.get(Calendar.DAY_OF_MONTH);
            monthParse = calendar.get(Calendar.MONTH)+1;
            yearParse = calendar.get(Calendar.YEAR);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * When displaying a class object, it displays the date
     * @return Returns a string that contains a date
     */
    @Override
    public String toString() { return day+"/"+month+"/"+year; }

    /**
     * Displays the date depending on the selected formatting
     * @param formatToDisplay Display format
     * @return Returns a string that contains a date
     */
    public String showDate(int formatToDisplay){
        Months mon = new Months(month, year);
        if(formatToDisplay == 1) return getWeekDayUsingModulo()+", "+day+" "+mon.getAllName()+" "+year;
        if(formatToDisplay == 2) return day+" "+mon.getAllName()+" "+year;
        if(formatToDisplay == 3) return day+"."+mon.getRomanName()+"."+year;
        if(formatToDisplay == 4) return getWeekDayUsingModulo().substring(0,2)+", "+day+"-"+mon.getShortName()+"-"+"2020";
        else return day+"/"+month+"/"+year;
    }

    /**
     * Main menu operation
     */
    public void mainMenu() throws IOException {
        Scanner scan = new Scanner(System.in);
        while (true) {
            System.out.println("\nEvents - 1\nExit - 0");
            int choice = scan.nextInt();
            if (choice == 1) {
                Events event = new Events(day,month, year);
                event.eventLog();
            }
            if (choice == 0) break;
        }
    }

    /**
     * Returns the name of the day of the week using Reference
     * @return day of the week name
     */
    public String getWeekDayUsingReference() {
        CalendarDate referenceDate = new CalendarDate(30, 11, 2020);
        String[] dayNames = {"Sunday","Monday","Tuesday","Wednesday", "Thursday","Friday","Saturday"};
        int numberOfWeekDay = 1;

        while (!(referenceDate.year == year)) {
            if(referenceDate.year > year) nextWeek();
            else backWeek();
        }
        while (!(referenceDate.month == month)){
            if(referenceDate.month > month) nextWeek();
            else backWeek();
        }
        while (referenceDate.day > day + 7 || referenceDate.day < day -7) {
            if(referenceDate.day > day) nextWeek();
            else backWeek();
        }
        int index = day - referenceDate.day + numberOfWeekDay;
        if(index < 0) index += 7;
        if(index > 6) index -= 7;

        return dayNames[index];
    }

    /**
     * Returns the name of the day of the week using Modulo
     * @return day of the week name
     */
    public String getWeekDayUsingModulo() {
        String[] dayNames = {"Saturday","Sunday","Monday","Tuesday","Wednesday", "Thursday","Friday"};
        int d = day;
        int m = month;
        int y = year;

        if (m == 1) {
            m = 13;
            y--;
        }
        if (m == 2) {
            m = 14;
            y--;
        }
        int K = y % 100;
        int J = y / 100;
        int x = (d + 13 * (m + 1) / 5 + K + K / 4 + J / 4 + 5 * J) % 7;
        return dayNames[x];
    }

    /**
     * Moves the date forward one week
     */
    public void nextWeek() {
        Months mon = new Months(month, year);
        int daysInMonth = mon.getMonthRange();
        day += 7;
        if (day > daysInMonth) {
            month++;
            day -= daysInMonth;
        }
        if (month > 12) {
            month = 1;
            year++;
        }
    }

    /**
     * Moves the date back one week
     */
    public void backWeek() {
        Months mon = new Months(month, year);
        day -= 7;
        if (day < 1) {
            month--;
            if (month < 1) {
                month = 12;
                year--;
            }
            day += mon.getBackMonthRange();
        }
    }
}