module org.example {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.mail;
    requires activation;

    opens org.example to javafx.fxml;
    exports org.example;
}