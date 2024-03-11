package code;

class ShoppingCartOperationTest {
    private static ShoppingCartOperation shoppingCart = new ShoppingCart();
    private static ShoppingCartOperationTest shoppingCartTest = new ShoppingCartOperationTest();

    public static void main(String[] args) {
        //shoppingCartTest.testAddProductMinusPrice();
        shoppingCartTest.testAddProductNameisNull();
    }

    public void testAddProductMinusPrice() {
        //Arrange
        String productName = "Product";
        int price = -10;
        int quantity = 10;

        // Act
        boolean result = shoppingCart.addProduct(productName, price, quantity);

        // Assert
        assert result == false : "Minus price";
    }

    public void testAddProductNameisNull() {
        //Arrange
        String productName = null;
        int price = 10;
        int quantity = 10;

        // Act
        boolean result = shoppingCart.addProduct(productName, price, quantity);

        // Assert
        assert result == false : "Name is null";
    }
}
