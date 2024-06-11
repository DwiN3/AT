package code;

import java.util.List;

public interface BookManagment {
    void addBook(String title, String author, int year);
    boolean removeBook(String title);
    List<Book> allBooks();
}
