package com.example.lab_9.ui.shoutbox;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.example.lab_9.MainMenuActivity;
import com.example.lab_9.SettingsActivity;
import com.example.lab_9.databinding.FragmentShoutboxBinding;

public class ShoutboxFragment extends Fragment {

    private FragmentShoutboxBinding binding;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        ShoutboxViewModel shoutboxViewModel =
                new ViewModelProvider(this).get(ShoutboxViewModel.class);

        binding = FragmentShoutboxBinding.inflate(inflater, container, false);
        View root = binding.getRoot();

        return root;
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}