package code;

public class Tables {
    Integer max(Integer[] digits) {
        if (digits == null || digits.length == 0) {
            return null;
        }

        Integer max = digits[0];
        for (int i = 0; i < digits.length; i++) {
            if (digits[i] != null && digits[i] > max) {
                max = digits[i];
            }
        }
        return max;
    }

    public void testMaxReturnsNullForEmptyInput() {
        // Given
        Integer[] input = {};

        // When
        Integer result = new Tables().max(input);

        // Then
        assert result == null : "Test max(null) should get null - 1";
    }

    public void testMaxReturnsNullForNullInput() {
        // Given
        Integer[] input = null;

        // When
        Integer result = new Tables().max(input);

        // Then
        assert result == null : "Test max(null) should get null -2";
    }

    public void testMaxReturnsValueForNullValueInput() {
        // Given
        Integer[] input = {1};

        // When
        Integer result = new Tables().max(input);

        // Then
        assert result == 1 : "Test max(null) should get null - 3";
    }

    public void testMaxReturnsCorrectMaxValue() {
        // Given
        Integer[] input = {1, 2, 3};

        // When
        Integer result = new Tables().max(input);

        // Then
        assert result == 3 : "Test max(null) should get null - 4";
    }

    public static void main(String[] args) {
        Tables tests = new Tables();
        tests.testMaxReturnsNullForEmptyInput();
        tests.testMaxReturnsNullForNullInput();
        tests.testMaxReturnsValueForNullValueInput();
        tests.testMaxReturnsCorrectMaxValue();
    }
}
