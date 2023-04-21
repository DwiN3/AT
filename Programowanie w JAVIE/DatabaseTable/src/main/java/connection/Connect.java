package connection;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.*;

public class Connect {
    String fileContent = new String(Files.readAllBytes(Paths.get("password.txt")), StandardCharsets.UTF_8);
    String[] lines = fileContent.split("\\r?\\n");
    private String driver = lines[0];
    private String host = lines[1];
    private String port = lines[2];
    private String dbname = lines[3];
    private String user = lines[4];
    private String password = lines[5];;

    private Connection connection;

    public Connect() throws IOException {
        connection = makeConnection();
    }

    public Connection getConnection() {
        return connection;
    }

    public void close() {
        try {
            connection.close();
        } catch (SQLException sqle) {
            System.err.println("Blad przy zamykaniu polaczenia: " + sqle);
        }
    }

    public Connection makeConnection() {
        try {
            Class.forName(driver);
            Connection connection = DriverManager.getConnection("jdbc:postgresql://" + host + ":" + port + "/" + dbname, user, password);
            return connection;
        } catch (ClassNotFoundException cnfe) {
            System.err.println("Blad ladowania sterownika: " + cnfe);
            return null;
        } catch (SQLException sqle) {
            System.err.println("Blad przy nawiązywaniu polaczenia: " + sqle);
            return null;
        }
    }
}
