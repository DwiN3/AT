package org.example;

import Bases.*;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;

import java.io.IOException;

public class DatabaseTableController {
    @FXML
    private TableView<Object> table;
    @FXML
    private TableColumn<Object, String> firstColumnTable, secondColumnTable, thirdColumnTable, fourthColumnTable;
    @FXML
    private Button buttonStudent, buttonOceny, buttonAdresy, buttonProwadzacy, buttonFakultety, buttonKierunkiStudiow;
    @FXML
    private Label textSelect;
    private String[] selectedMode = {"Student", "Oceny", "Adresy", "Prowadzacy", "Fakultety", "Kierunki Studiow"};
    private String[] firstColumnController, secondColumnController, thirdColumnController, fourthColumnController;
    private int option = 0;

    @FXML
    private void initialize() {
        buttonStudent.setOnAction(event -> {
            option = 0;
            try {
                setButton();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        buttonOceny.setOnAction(event -> {
            option = 1;
            try {
                setButton();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        buttonAdresy.setOnAction(event -> {
            option = 2;
            try {
                setButton();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        buttonProwadzacy.setOnAction(event -> {
            option = 3;
            try {
                setButton();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        buttonFakultety.setOnAction(event -> {
            option = 4;
            try {
                setButton();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        buttonKierunkiStudiow.setOnAction(event -> {
            option = 5;
            try {
                setButton();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    private void setButton() throws IOException {
        table.getColumns().clear();
        makeTabel();
        textSelect.setText("Wybrano: "+selectedMode[option]);
    }

    public void makeTabel() throws IOException {
        Management management = new Management(option);
        firstColumnController = management.getColumn1();
        secondColumnController = management.getColumn2();
        thirdColumnController = management.getColumn3();
        fourthColumnController = management.getColumn4();

        firstColumnTable.setText(management.getNameColumn1());
        secondColumnTable.setText(management.getNameColumn2());
        thirdColumnTable.setText(management.getNameColumn3());
        fourthColumnTable.setText(management.getNameColumn4());

        firstColumnTable.setCellValueFactory(new PropertyValueFactory<>(management.getText1()));
        secondColumnTable.setCellValueFactory(new PropertyValueFactory<>(management.getText2()));
        thirdColumnTable.setCellValueFactory(new PropertyValueFactory<>(management.getText3()));
        fourthColumnTable.setCellValueFactory(new PropertyValueFactory<>(management.getText4()));

        table.getColumns().addAll(firstColumnTable, secondColumnTable, thirdColumnTable, fourthColumnTable);

        table.getItems().clear();

        Object object = new Object();
        for (int n = 0; n < firstColumnController.length; n++) {
            switch (option) {
                case 0:
                    Student student = new Student(firstColumnController[n], secondColumnController[n], thirdColumnController[n], fourthColumnController[n]);
                    object = student;
                    break;
                case 1:
                    Oceny oceny = new Oceny(firstColumnController[n], secondColumnController[n], thirdColumnController[n], fourthColumnController[n]);
                    object = oceny;
                    break;
                case 2:
                    Adresy adresy = new Adresy(firstColumnController[n], secondColumnController[n], thirdColumnController[n], fourthColumnController[n]);
                    object = adresy;
                    break;
                case 3:
                    Prowadzacy prowadzacy = new Prowadzacy(firstColumnController[n], secondColumnController[n], thirdColumnController[n], fourthColumnController[n]);
                    object = prowadzacy;
                    break;
                case 4:
                    Fakultety fakultety = new Fakultety(firstColumnController[n], secondColumnController[n], thirdColumnController[n], fourthColumnController[n]);
                    object = fakultety;
                    break;
                case 5:
                    KierunkiStudiow kierunkiStudiow = new KierunkiStudiow(firstColumnController[n], secondColumnController[n], thirdColumnController[n], fourthColumnController[n]);
                    object = kierunkiStudiow;
                    break;
                default:
                    break;
            }
            table.getItems().add(object);
        }
    }
}