package Other;

import Retrofit.Models.Flashcard;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * The class that is responsible for setting up the game
 */
public class SetGame {
    private DateInstance dateInstance = DateInstance.getInstance();
    private String[] NameWord, correctANS, sentense, sentenseTra;
    private String[] ans1, ans2, ans3, ans4;
    private int listSize = 0, borrder = 0;
    private ArrayList<Flashcard> wordsList;

    /**
     * Constructs a SetGame object
     * @param data         the data
     * @param mode         the mode
     * @param language     the language
     * @param wordsListAll the list of all flashcards
     */
    public SetGame(String data, String mode, String language, ArrayList<Flashcard> wordsListAll) {
        borrder = dateInstance.getBorderMaxFlashcards();
        if (mode.equals("quiz")) {
            Random randomWords = new Random();
            Set<Integer> selectedIndices = new HashSet<>();
            ArrayList<Flashcard> selectedWords = new ArrayList<>();
            if (wordsListAll.size() <= borrder) borrder = wordsListAll.size();

            while (selectedIndices.size() < borrder) {
                int randomIndex = randomWords.nextInt(wordsListAll.size());

                if (!selectedIndices.contains(randomIndex)) {
                    selectedIndices.add(randomIndex);
                    selectedWords.add(wordsListAll.get(randomIndex));
                }
            }
            wordsList = selectedWords;
        } else wordsList = wordsListAll;

        this.NameWord = new String[wordsList.size()];
        this.correctANS = new String[wordsList.size()];
        this.sentense = new String[wordsList.size()];
        this.sentenseTra = new String[wordsList.size()];
        this.ans1 = new String[wordsList.size()];
        this.ans2 = new String[wordsList.size()];
        this.ans3 = new String[wordsList.size()];
        this.ans4 = new String[wordsList.size()];

        Random random = new Random();
        for (int i = 0; i < wordsList.size(); i++) {
            if (language.equals("pl")) {
                NameWord[i] = wordsList.get(i).getWord();
                correctANS[i] = wordsList.get(i).getTranslatedWord();
            } else {
                NameWord[i] = wordsList.get(i).getTranslatedWord();
                correctANS[i] = wordsList.get(i).getWord();
            }
            sentense[i] = wordsList.get(i).getExample();
            sentenseTra[i] = wordsList.get(i).getTranslatedExample();

            if (mode.equals("quiz")) {
                Set<String> uniqueWords = new HashSet<>();
                uniqueWords.add(correctANS[i]);

                while (uniqueWords.size() < 4) {
                    int randomIndex = random.nextInt(wordsList.size());
                    String randomWord;
                    if (language.equals("pl")) randomWord = wordsList.get(randomIndex).getTranslatedWord();
                    else randomWord = wordsList.get(randomIndex).getWord();
                    uniqueWords.add(randomWord);
                }

                String[] uniqueWordsArray = uniqueWords.toArray(new String[0]);
                ans1[i] = uniqueWordsArray[0];
                ans2[i] = uniqueWordsArray[1];
                ans3[i] = uniqueWordsArray[2];
                ans4[i] = uniqueWordsArray[3];
            }
        }
        listSize = wordsList.size();
    }

    /**
     * Returns the name of the word at the specified index.
     * @param i the index of the word
     * @return the name of the word at the specified index
     */
    public String getNameWord(int i) {
        return NameWord[i];
    }

    /**
     * Returns the first answer option at the specified index.
     * @param i the index of the answer option
     * @return the first answer option at the specified index
     */
    public String getAns1(int i) {
        return ans1[i];
    }

    /**
     * Returns the second answer option at the specified index.
     * @param i the index of the answer option
     * @return the second answer option at the specified index
     */
    public String getAns2(int i) {
        return ans2[i];
    }

    /**
     * Returns the third answer option at the specified index.
     * @param i the index of the answer option
     * @return the third answer option at the specified index
     */
    public String getAns3(int i) {
        return ans3[i];
    }

    /**
     * Returns the fourth answer option at the specified index.
     * @param i the index of the answer option
     * @return the fourth answer option at the specified index
     */
    public String getAns4(int i) {
        return ans4[i];
    }

    /**
     * Returns the correct answer at the specified index.
     * @param i the index of the correct answer
     * @return the correct answer at the specified index
     */
    public String getCorrectANS(int i) {
        return correctANS[i];
    }

    /**
     * Returns the sentence at the specified index.
     * @param i the index of the sentence
     * @return the sentence at the specified index
     */
    public String getSentense(int i) {
        return sentense[i];
    }

    /**
     * Returns the translated sentence at the specified index.
     * @param i the index of the translated sentence
     * @return the translated sentence at the specified index
     */
    public String getSentenseTra(int i) {
        return sentenseTra[i];
    }

    /**
     * Returns the size of the word list.
     * @return the size of the word list
     */
    public int getListSize() {
        return listSize;
    }

    /**
     * Returns the border value.
     * @return the border value
     */
    public int getBorrder() {
        return borrder;
    }
}