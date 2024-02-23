package org.example;

public class Tables {
    Integer max(Integer[] digits) {
        if (digits == null || digits.length == 0) {
            return null;
        }

        Integer max = digits[0];
        for (Integer digit : digits) {
            if (digit != null && digit > max) {
                max = digit;
            }
        }
        return max;
    }

    public static void main(String[] args) {
        assert (new Tables().max(null) == null) : "Test max(null) should get null";
        assert (new Tables().max(new Integer[]{}) == null) : "Test max(null) should get null";
        assert (new Tables().max(new Integer[]{1}) == 1) : "Test max(null) should get null";
        assert (new Tables().max(new Integer[]{1,2,3}) == 3) : "Test max(null) should get null";
    }
}
