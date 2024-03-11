package code;

import java.util.List;

public class ShoppingCart implements ShoppingCartOperation {
    @Override
    public boolean addProduct(String productName, int price, int quantity) {
        if (price <= 0 || quantity <= 0) {
            return false;
        }

        return true;
    }
}
