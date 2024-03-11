package code;

import java.util.List;
public interface ShoppingCartOperation {
    // Dodawanie produktu do koszyka
    boolean addProduct(String productName, int price, int quantity);
    // Usuwanie produktu z koszyka
}
