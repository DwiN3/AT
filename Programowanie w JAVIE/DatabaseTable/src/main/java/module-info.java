module org.example {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.sql;

    opens org.example to javafx.fxml;
    exports org.example;
    exports connection;
    opens connection to javafx.fxml;
    exports Bases;
    opens Bases to javafx.fxml;
}