package org.example;

import java.io.*;
import java.time.LocalDate;
import java.util.Scanner;

/**
 * The class supports write, read, event cleanup options
 *
 * @author Kamil Dereń
 */
public class Events {
    private int day, month, year; // Variables storing day, month and year

    /**
     * The constructor sets the date
     * @param day_ Sets the day
     * @param month_ Sets the month
     * @param year_ Sets the year
     */
    public Events(int day_, int month_, int year_){
        day = day_;
        month = month_;
        year =  year_;
    }

    /**
     * Main event service menu
     */
    public void eventLog() throws IOException {
        Scanner scan = new Scanner(System.in);
        while(true) {
            System.out.println("\nShow events - 1\nSave new Event - 2\nClear all Events - 3\nExit - 0");
            int choice = scan.nextInt();
            if (choice == 1) showEvents();
            if (choice == 2) newEvent();
            if (choice == 3) clearAllEvents();
            if (choice == 0) break;
        }
    }

    /**
     * Shows all events
     */
    public void showEvents(){
        File file = new File("./files/events.txt");
        if(file.length() != 0) System.out.println("\nEvents: ");
        else System.out.println("\nEmpty events");
        try( BufferedReader br = new BufferedReader(new FileReader(file))){
            PrintWriter pw = new PrintWriter(System.out, true);
            String s;
            while ( (s = br.readLine()) != null )
                pw.println(s);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Creating a new event
     */
    public void newEvent() throws IOException {
        String data, info;
        int dayEvent, monthEvent, yearEvent, choice;
        LocalDate currentdate = LocalDate.now();
        this.day = currentdate.getDayOfMonth();
        this.month = currentdate.getMonthValue();
        this.year = currentdate.getYear();

        System.out.println("Enter date (1 - Manually / 2 - Automatically)");
        Scanner scan = new Scanner(System.in);
        choice = scan.nextInt();
        if(choice == 1){
            System.out.println("Day: ");
            dayEvent = scan.nextInt();
            System.out.println("Month: ");
            monthEvent = scan.nextInt();
            System.out.println("Year: ");
            yearEvent = scan.nextInt();
            data = dayEvent+"/"+monthEvent+"/"+yearEvent;
        }
        else data = day+"/"+month+"/"+year;

        System.out.println("Provide information about the event: ");
        scan.nextLine();
        info = scan.nextLine();

        File file = new File("./files/events.txt");
        Writer eventFile = new BufferedWriter(new FileWriter(file, true));
        if(file.length() != 0) eventFile.append("\n"+data+"   "+info);
        else eventFile.append(data+"   "+info);
        eventFile.close();
    }

    /**
     * Clears all events
     */
    public void clearAllEvents(){
        try (PrintWriter writer = new PrintWriter("./files/events.txt")) {
            writer.print("");
            System.out.println("All events have been removed");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}
