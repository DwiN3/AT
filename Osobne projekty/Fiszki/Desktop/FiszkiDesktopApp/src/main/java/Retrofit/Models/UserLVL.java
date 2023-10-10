package Retrofit.Models;

import com.google.gson.annotations.SerializedName;

/**
 * Model class representing user level information
 */
public class UserLVL {
    private int result, level, requiredPoints, points;
    private String content;

    @SerializedName("body")
    private String text;

    /**
     * Constructs a UserLVL object with the result value
     * @param result the result value
     */
    public UserLVL(int result) {
        this.result = result;
    }

    /**
     * Returns the result value
     * @return the result value
     */
    public int getResult() { return result; }

    /**
     * Returns the content
     * @return the content
     */
    public String getContent() { return content; }

    /**
     * Sets the result value
     * @param result the result value to set
     */
    public void setResult(int result) { this.result = result; }

    /**
     * Returns the level
     * @return the level
     */
    public int getLevel() { return level; }

    /**
     * Returns the required points for the level
     * @return the required points for the level
     */
    public int getRequiredPoints() { return requiredPoints; }

    /**
     * Returns the points
     * @return the points
     */
    public int getPoints() { return points; }

    /**
     * Returns the text
     * @return the text
     */
    public String getText() {return text; }
}
