package com.kdbk.fiszki.Other;

import android.content.Context;
import android.content.Intent;

public class NextActivity {
    private final Context context;

    public NextActivity(Context context) {
        this.context = context;
    }

    public void openActivity(Class<?> toClass) {
        Intent intent = new Intent(context, toClass);
        context.startActivity(intent);
    }

    public void openActivity(Class<?> toClass, Intent intent) {
        intent.setClass(context, toClass);
        context.startActivity(intent);
    }
}
