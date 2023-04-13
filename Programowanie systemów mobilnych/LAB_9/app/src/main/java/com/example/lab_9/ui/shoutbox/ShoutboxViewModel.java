package com.example.lab_9.ui.shoutbox;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class ShoutboxViewModel extends ViewModel {

    private final MutableLiveData<String> mText;

    public ShoutboxViewModel() {
        mText = new MutableLiveData<>();
        mText.setValue("This is home fragment");
    }

    public LiveData<String> getText() { return mText; }
}