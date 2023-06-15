package Other;

/**
 * The DateInstance class represents a single instance that stores various data related to the game session and user information
 */
public class DateInstance {
    private static DateInstance instance = null;

    // Token
    private static String userName ="", token="";

    // Game
    private static String gameMode = "", categoryName = "", language ="", selectData ="";
    private static int bestTrain = 0, points=0, allWords=0, borderMaxFlashcards =15, borderMinFlashcardQuiz=10;

    /**
     * Returns the singleton instance of DateInstance
     * @return The singleton instance of DateInstance
     */
    public static DateInstance getInstance() {
        if (instance == null) {
            instance = new DateInstance();
        }
        return instance;
    }


    /**
     * Retrieves the userName
     * @return The user name.
     */
    public String getUserName() {
        return userName;
    }

    /**
     * Sets the userName
     * @param userName The userName to be set
     */
    public void setUserName(String userName) {
        DateInstance.userName = userName;
    }

    /**
     * Retrieves the token
     * @return The token
     */
    public String getToken() {
        return token;
    }

    /**
     * Sets the token
     * @param token The token to be set
     */
    public void setToken(String token) {
        DateInstance.token = token;
    }

    /**
     * Retrieves the current game mode
     * @return The current game mode
     */
    public String getGameMode() {
        return gameMode;
    }

    /**
     * Sets the game mode
     * @param gameMode The game mode to be set
     */
    public void setGameMode(String gameMode) {
        DateInstance.gameMode = gameMode;
    }

    /**
     * Retrieves the name of the category
     * @return The name of the category
     */
    public String getCategoryName() {
        return categoryName;
    }

    /**
     * Sets the name of the category
     * @param nameCategory The name of the category to be set
     */
    public void setCategoryName(String nameCategory) {
        DateInstance.categoryName = nameCategory;
    }

    /**
     * Retrieves the language
     * @return The language
     */
    public String getLanguage() {
        return language;
    }

    /**
     * Sets the language
     * @param language The language to be set
     */
    public void setLanguage(String language) {
        DateInstance.language = language;
    }

    /**
     * Retrieves the selected data
     * @return The selected data
     */
    public String getSelectData() {
        return selectData;
    }

    /**
     * Sets the selected data
     * @param selectData The selected data to be set
     */
    public void setSelectData(String selectData) {
        DateInstance.selectData = selectData;
    }

    /**
     * Retrieves the points
     * @return The points
     */
    public int getPoints() {
        return points;
    }

    /**
     * Sets the points
     * @param points The points to be set
     */
    public void setPoints(int points) {
        DateInstance.points = points;
    }

    /**
     * Retrieves the total number of words
     * @return The total number of words
     */
    public int getAllWords() {
        return allWords;
    }

    /**
     * Sets the total number of words
     * @param allWords The total number of words to be set
     */
    public void setAllWords(int allWords) {
        DateInstance.allWords = allWords;
    }

    /**
     * Retrieves the best training score
     * @return The best training score
     */
    public int getBestTrain() {
        return bestTrain;
    }

    /**
     * Sets the best training score
     * @param bestTrain The best training score to be set
     */
    public void setBestTrain(int bestTrain) {
        DateInstance.bestTrain = bestTrain;
    }

    /**
     * Retrieves the maximum number of flashcards for the border
     * @return The maximum number of flashcards for the border
     */
    public int getBorderMaxFlashcards() {
        return borderMaxFlashcards;
    }

    /**
     * Sets the maximum number of flashcards for the border
     * @param borderMaxFlashcards The maximum number of flashcards for the border to be set
     */
    public void setBorderMaxFlashcards(int borderMaxFlashcards) { DateInstance.borderMaxFlashcards = borderMaxFlashcards; }

    /**
     * Retrieves the minimum number of flashcards for the quiz border
     * @return The minimum number of flashcards for the quiz border
     */
    public int getBorderMinFlashcardQuiz() {
        return borderMinFlashcardQuiz;
    }

    /**
     * Sets the minimum number of flashcards for the quiz border
     * @param borderMinFlashcardQuiz The minimum number of flashcards for the quiz border to be set
     */
    public void setBorderMinFlashcardQuiz(int borderMinFlashcardQuiz) { DateInstance.borderMinFlashcardQuiz = borderMinFlashcardQuiz; }
}
