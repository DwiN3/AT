package org.example;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class SenderController {

    @FXML
    private TextField textMail;
    @FXML
    private TextField textSubject;
    @FXML
    private TextField textMessage;
    @FXML
    private Label textFileInfo;
    @FXML
    private Button sendButton;
    private String mail, subject, message, filePath = "";

    public String getMail(){
        return mail;
    }
    public String getSubject(){
        return subject;
    }
    public String getMessage(){
        return message;
    }

    @FXML
    private void pressToSend() throws IOException {
        mail = textMail.getText();
        subject = textSubject.getText();
        message = textMessage.getText();
        System.out.println("Mail: "+mail+"\nSubject: "+subject+"\n Message: "+message);
        SendEmailTLS sendEmailTLS = new SendEmailTLS(mail, subject, message, filePath);
        sendEmailTLS.sendMessage();
    }

    @FXML
    private void addFile() {
        FileChooser fc = new FileChooser();
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Image files","*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"));
        List<File> f = fc.showOpenMultipleDialog(null);
        for (File file: f) {
            filePath = file.getAbsolutePath();
            textFileInfo.setText("Photo selected");
            textFileInfo.setTextFill(Color.GREEN);
        }
    }

    @FXML
    private void delFile() {
        filePath = "";
        textFileInfo.setText("No photo selected");
        textFileInfo.setTextFill(Color.RED);
    }
}
