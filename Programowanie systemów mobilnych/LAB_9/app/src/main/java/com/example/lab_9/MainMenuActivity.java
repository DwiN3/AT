package com.example.lab_9;

import android.content.Context;
import android.content.Intent;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.text.Spannable;
import android.text.SpannableStringBuilder;
import android.text.style.StyleSpan;
import android.view.Menu;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.example.lab_9.databinding.ActivityMainMenuBinding;
import com.google.android.material.navigation.NavigationView;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class MainMenuActivity extends AppCompatActivity implements SelectListener{
    private AppBarConfiguration mAppBarConfiguration;
    private ActivityMainMenuBinding binding;
    private RecyclerView mRecycleView;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager mLayoutManager;
    private String actualLogin, content, login, date, id, message;
    private TextView TextViewAddMessages, TextMessage, TextLogin;
    private ImageButton buttonMessage;
    private SwipeRefreshLayout swipeRefreshLayout;
    private boolean choiceToLimitMessage;
    ArrayList<MessageList> messageLists = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainMenuBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.appBarMainMenu.toolbar);

        DrawerLayout drawer = binding.drawerLayout;
        NavigationView navigationView = binding.navView;

        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        mAppBarConfiguration = new AppBarConfiguration.Builder(
                R.id.nav_shoutbox, R.id.nav_gallery, R.id.nav_settings)
                .setOpenableLayout(drawer)
                .build();
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main_menu);
        NavigationUI.setupActionBarWithNavController(this, navController, mAppBarConfiguration);
        NavigationUI.setupWithNavController(navigationView, navController);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main_menu, menu);
        setID();
        setValues();
        deleteMessageUsingSwipe();

        buttonMessage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(!checkInternetConnection()) openSettings();
                message = TextMessage.getText().toString();
                TextMessage.setText("");
                createMessage();
            }
        });
        swipeRefreshLayout.setOnRefreshListener(new SwipeRefreshLayout.OnRefreshListener() {
            @Override
            public void onRefresh() {
                if(!checkInternetConnection()) openSettings();
                displayMessages();
                swipeRefreshLayout.setRefreshing(false);
            }
        });
        contentRefresh60s();
        return true;
    }

    @Override
    public boolean onSupportNavigateUp() {
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main_menu);
        return NavigationUI.navigateUp(navController, mAppBarConfiguration)
                || super.onSupportNavigateUp();
    }

    public void setValues(){
        Intent intent = getIntent();
        actualLogin = intent.getStringExtra("Login");
        choiceToLimitMessage = intent.getExtras().getBoolean("Choice");
        SpannableStringBuilder formatDisplayLogin = new SpannableStringBuilder("User logged:   "+actualLogin);
        StyleSpan bss = new StyleSpan(android.graphics.Typeface.BOLD);
        formatDisplayLogin.setSpan(bss, 14, 15+actualLogin.length(), Spannable.SPAN_INCLUSIVE_INCLUSIVE);
        TextLogin.setText(formatDisplayLogin);
    }

    public void displayMessages(){
        messageLists.clear();
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://tgryl.pl/shoutbox/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonPlaceholderAPI jsonPlaceholderAPI = retrofit.create(JsonPlaceholderAPI.class);
        Call<List<Post>> call = jsonPlaceholderAPI.getPostsLast8();
        if(choiceToLimitMessage) call = jsonPlaceholderAPI.getPostsAll();

        call.enqueue(new Callback<List<Post>>() {
            @RequiresApi(api = Build.VERSION_CODES.O)
            @Override
            public void onResponse(Call<List<Post>> call, Response<List<Post>> response) {
                if(!response.isSuccessful()){
                    Toast.makeText(MainMenuActivity.this,"Code: "+response.code(), Toast.LENGTH_SHORT).show();
                }
                List<Post> posts = response.body();
                for(Post post: posts){
                    login = post.getLogin();
                    content = post.getContent();
                    id = post.getId();
                    Instant instant = Instant.parse(post.getDate());
                    DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd.MM.yyyy HH:mm:ss");
                    String formattedDate = instant.atZone(ZoneId.systemDefault()).format(formatter);
                    date = formattedDate;
                    TextViewAddMessages.append("Login: "+login+"\nData: "+date+"\nContent: "+content);
                    messageLists.add(new MessageList(login,date,content, id));
                }
            }
            @Override
            public void onFailure(Call<List<Post>> call, Throwable t) {}
        });
        mRecycleView.setHasFixedSize(true);
        mLayoutManager = new LinearLayoutManager(this);
        mAdapter = new MessageAdapter(messageLists,this);
        mRecycleView.setLayoutManager(mLayoutManager);
        mRecycleView.setAdapter(mAdapter);
    }

    public void createMessage(){
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://tgryl.pl/shoutbox/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonPlaceholderAPI jsonPlaceholderAPI = retrofit.create(JsonPlaceholderAPI.class);
        Post post = new Post(actualLogin,message);
        Call<List<Post>> call = jsonPlaceholderAPI.createMessage(post);

        call.enqueue(new Callback<List<Post>>() {
            @Override
            public void onResponse(Call<List<Post>> call, Response<List<Post>> response) {
                if(!response.isSuccessful()){
                    if(response.code() == 400) Toast.makeText(MainMenuActivity.this,"The message is empty", Toast.LENGTH_SHORT).show();
                    else Toast.makeText(MainMenuActivity.this,"Code: "+response.code(), Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<List<Post>> call, Throwable t) {}
        });
    }

    public void deleteMessage(String messageID){
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://tgryl.pl/shoutbox/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        JsonPlaceholderAPI jsonPlaceholderAPI = retrofit.create(JsonPlaceholderAPI.class);
        Call<List<Post>> call = jsonPlaceholderAPI.deletePost(messageID);

        call.enqueue(new Callback<List<Post>>() {
            @Override
            public void onResponse(Call<List<Post>> call, Response<List<Post>> response) {
                Toast.makeText(MainMenuActivity.this,"Code: "+response.code(), Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onFailure(Call<List<Post>> call, Throwable t) {}
        });
    }

    public void deleteMessageUsingSwipe(){
        ItemTouchHelper.SimpleCallback simpleItemTouchCallback = new ItemTouchHelper.SimpleCallback(0, ItemTouchHelper.LEFT | ItemTouchHelper.RIGHT) {
            @Override
            public boolean onMove(@NonNull RecyclerView recyclerView, @NonNull RecyclerView.ViewHolder viewHolder, @NonNull RecyclerView.ViewHolder target) {
                return false;
            }

            @Override
            public void onSwiped(@NonNull RecyclerView.ViewHolder viewHolder, int direction) {
                if(!checkInternetConnection()) openSettings();
                int position = viewHolder.getAdapterPosition();

                Retrofit retrofit = new Retrofit.Builder()
                        .baseUrl("https://tgryl.pl/shoutbox/")
                        .addConverterFactory(GsonConverterFactory.create())
                        .build();
                JsonPlaceholderAPI jsonPlaceholderAPI = retrofit.create(JsonPlaceholderAPI.class);
                Call<List<Post>> call = jsonPlaceholderAPI.getPostsLast8();
                if(choiceToLimitMessage) call = jsonPlaceholderAPI.getPostsAll();
                call.enqueue(new Callback<List<Post>>() {
                    @RequiresApi(api = Build.VERSION_CODES.O)
                    @Override
                    public void onResponse(Call<List<Post>> call, Response<List<Post>> response) {
                        if(!response.isSuccessful()){
                            Toast.makeText(MainMenuActivity.this,"Code: "+response.code(), Toast.LENGTH_SHORT).show();
                        }
                        List<Post> posts = response.body();
                        int licznik = 0;
                        for(Post post : posts){
                            if(licznik == position){
                                login = post.getLogin();
                                if(actualLogin.equals(login)){
                                    id = post.getId();
                                    deleteMessage(id);
                                    mAdapter.notifyItemRemoved(position);
                                    messageLists.remove(position);
                                    displayMessages();
                                }
                                else{
                                    displayMessages();
                                    Toast.makeText(MainMenuActivity.this,"You cannot modify someone's message", Toast.LENGTH_SHORT).show();
                                }
                            } licznik++;
                        }
                    }
                    @Override
                    public void onFailure(Call<List<Post>> call, Throwable t) {}
                });
            }
        };
        ItemTouchHelper itemTouchHelper = new ItemTouchHelper(simpleItemTouchCallback);
        itemTouchHelper.attachToRecyclerView(mRecycleView);
    }

    public void contentRefresh60s(){
        int count = 0;
        count++;
        displayMessages();
        refresh(60000);
    }
    private void refresh(int milliseconds){
        final Handler handler = new Handler();
        final Runnable runnable = new Runnable() {
            @Override
            public void run() { contentRefresh60s(); }
        };
        handler.postDelayed(runnable, milliseconds);
    }

    @Override
    public void onItemClicked(MessageList messageList) {
        if(!checkInternetConnection()) openSettings();
        else{
            Intent intent = new Intent(this, EditMessageActivity.class);
            intent.putExtra("ActualLogin", actualLogin);
            intent.putExtra("EditLogin", messageList.geteName());
            intent.putExtra("EditDate", messageList.geteDate());
            intent.putExtra("EditContent", messageList.geteMessage());
            intent.putExtra("EditID", messageList.geteID());
            intent.putExtra("Choice", choiceToLimitMessage);
            startActivity(intent);
        }
    }

    public boolean checkInternetConnection() {
        ConnectivityManager connectivityManager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
        if(activeNetworkInfo != null && activeNetworkInfo.isConnected()) return true;
        else return false;
    }

    public void openSettings(){
        Intent intent = new Intent(this, SettingsActivity.class);
        startActivity(intent);
    }

    public void setID(){
        TextViewAddMessages = findViewById(R.id.id_textViewAddMessages);
        mRecycleView = findViewById(R.id.id_recyclerView);
        TextMessage = findViewById(R.id.id_message);
        buttonMessage = findViewById(R.id.id_messageButton);
        TextLogin = findViewById(R.id.id_login);
        swipeRefreshLayout = findViewById(R.id.id_swipeRefreshLayout);
    }
}
