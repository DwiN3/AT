package code;

import java.util.List;

public class Library {
    private final BookManagment bookManagment;
    public Library(BookManagment bookManagment){
        this.bookManagment = bookManagment;
    }
    public boolean borrowBook(String title){
        return bookManagment.removeBook(title);
    }
    public void returnBook(String title, String author, int year){
        bookManagment.addBook(title, author, year);
    }
    public List<Book> books(){
        return bookManagment.allBooks();
    }
}