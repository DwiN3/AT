package com.kdbk.fiszki.Instance;

public class FlashcardInfoInstance {
    private static FlashcardInfoInstance instance = null;
    private static String nameCollection ="", id_word="";


    public static FlashcardInfoInstance getInstance() {
        if (instance == null) {
            instance = new FlashcardInfoInstance();
        }
        return instance;
    }

    public String getNameCollection() {
        return nameCollection;
    }

    public void setNameCollection(String nameCollection) { FlashcardInfoInstance.nameCollection = nameCollection; }

    public void setId_word(String id_word) {
        FlashcardInfoInstance.id_word = id_word;
    }

    public String getId_word() {return id_word;}
}
