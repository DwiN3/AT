module org.example {
    requires javafx.controls;
    requires javafx.fxml;

    opens app to javafx.fxml;
    exports app;
    exports Controllers;
    opens Controllers to javafx.fxml;
}