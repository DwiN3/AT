/**
 * The class that contains numbers, ranges, and month names
 *
 * @author Kamil Dereń
 */
class Months{
    private int ranges[] = {31,28,31,30,31,30,31,31,30,31,30,31}, range, number; // An array containing a range of months
    private String allNames[] = {"January","February","March","April","May","June","July","August","September","November","October","December"}, allName; // An array containing the names of the month
    private String romanNames[] = {"I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"}, romanName; // An array containing the roman names of the month
    private String shortNames[] = {"JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"}, shortName; // An array containing the short names of the month
    private int year; //Selected year

    /**
     * Stores the entire month name
     * @return Returns the name of the stored month
     */
    public String getAllName(){ return allName; }

    /**
     * Stores the Roman name of the month
     * @return Returns the name of the stored month
     */
    public String getRomanName(){ return romanName; }

    /**
     * Stores the abbreviated name of the month
     * @return Returns the name of the stored month
     */
    public String getShortName(){ return shortName; }

    /**
     * A function to return a range for the month
     * @return Returns the current range
     */
    public int getMonthRange(){ return range; }

    /**
     * The function checks the range value for the previous month
     * @return Returns the range of the previous month
     */
    public int getBackMonthRange(){
        if(leapYear() && number == 3){ return 29;}
        if(number == 1) return 31;
        else return ranges[number-2];
    }

    /**
     * Checks if the year is a leap year
     * @return Returns true if the year is a leap year or false if it is not a leap year
     */
    public boolean leapYear(){
        if(year % 4 ==0 && year % 100 !=0 || year % 400 == 0 ){
            return true;
        }
        else return false;
    }

    /**
     * Sets the number, range, and name for the selected month
     * @param _month Month typed
     * @param _year Year entered
     */
    public Months(int _month, int _year){
        this.number = _month;
        this.year = _year;
        this.allName = allNames[_month-1];
        this.shortName = shortNames[_month-1];
        this.romanName = romanNames[_month-1];
        if(leapYear() && _month == 2) range = 29;
        else range = ranges[_month-1];
    }
}