package org.example;

import java.util.ArrayList;
import java.util.Collections;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;

public class CalendarController {
    @FXML
    private Label textShowDate, textCollection;
    @FXML
    private TextField textGetDate;
    @FXML
    private Button buttonEnter;
    private CalendarDate cal;
    private int formatNumber=0;
    private ArrayList<CalendarDate> list = new ArrayList<CalendarDate>();
    private Months mon;

    private boolean checkDate(){
        if(textGetDate.getText().isEmpty()) return false;
        else return true;
    }

    private void setTextCollection(){
        if(checkDate()){
            String text = "";
            for (int n = 0; n < list.size(); n++) text += list.get(n) + "\n";
            textCollection.setText(text);
        }
    }

    private void setScreen(){
        if(formatNumber == 1 || formatNumber == 4) textShowDate.setText(cal.showDate(formatNumber));
        else textShowDate.setText(cal.showDate(formatNumber)+"\n"+cal.getWeekDayUsingModulo());
        if (mon.leapYear()) {
            String currentText = textShowDate.getText();
            textShowDate.setText(currentText + "\nRok przestępny");
        }
    }

    @FXML
    private void enter(){
        if(!checkDate()) cal = new CalendarDate();
        else cal = CalendarDate.parse(textGetDate.getText());
        mon = new Months(cal.getMonth(), cal.getYear());
        setScreen();
    }

    @FXML
    private void nextWeek(){
        if(checkDate()){
            cal.nextWeek();
            mon = new Months(cal.getMonth(), cal.getYear());
            setScreen();
        }
    }

    @FXML
    private void backWeek(){
        if(checkDate()){
            cal.backWeek();
            mon = new Months(cal.getMonth(), cal.getYear());
            setScreen();
        }
    }

    @FXML
    private void changeFormat(){
        if(checkDate()){
            formatNumber++;
            if(formatNumber>4) formatNumber=0;
            cal.showDate(formatNumber);
            setScreen();
        }
    }

    @FXML
    private void addToCollection(){
        if(checkDate()){
            CalendarDate calTMP = new CalendarDate(cal.getDay(), cal.getMonth(), cal.getYear());
            list.add(calTMP);
            setTextCollection();
        }
    }

    @FXML
    private void clearCollection(){
        if(checkDate()){
            list.clear();
            setTextCollection();
        }
    }

    @FXML
    private void sortCollection(){
        if(checkDate()){
            Collections.sort(list, new SortDate());
            setTextCollection();
        }
    }
}
