package connection;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.sql.*;
public class Connect {

    private String driver = "org.postgresql.Driver";

    private String host = "195.150.230.208";

    private String port = "5432";//wymagane kiedy nie jest domyślny dla bazy

    private String dbname = "2023_deren_kamil";
    private String user = "2023_deren_kamil";
    private File passwordFile = new File("password.txt");
    private String password = new String(Files.readAllBytes(passwordFile.toPath()));
    private String url = "jdbc:postgresql://" + host+":"+port + "/" + dbname; private String pass = password;
    private Connection connection;

    public Connect () throws IOException {
        connection = makeConnection(); }

    public Connection getConnection(){
        return(connection);
    }
    public void close() {
        try {

            connection.close(); }

        catch (SQLException sqle){
            System.err.println("Blad przy zamykaniu polaczenia: " + sqle);

        } }

    public Connection makeConnection(){
        try {
            Class.forName(driver);
            Connection connection = DriverManager.getConnection(url, user, pass); return(connection);

        }
        catch(ClassNotFoundException cnfe) {
            System.err.println("Blad ladowania sterownika: " + cnfe);

            return(null);
        }
        catch(SQLException sqle) {
            System.err.println("Blad przy nawiązywaniu polaczenia: " + sqle);

            return(null);
        }
    } }
