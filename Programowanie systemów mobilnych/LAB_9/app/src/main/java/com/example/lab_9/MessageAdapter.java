package com.example.lab_9;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;

public class MessageAdapter extends RecyclerView.Adapter<MessageAdapter.MyViewHolder>{

    public ArrayList<MessageList> mElementList;
    private SelectListener listener;

    public MessageAdapter(ArrayList<MessageList> elementArrayList, SelectListener listener){
        mElementList = elementArrayList;
        this.listener = listener;
    }

    @NonNull
    @Override
    public MyViewHolder onCreateViewHolder(@NonNull ViewGroup viewGroup, int i) {
        View v = LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.message_list, viewGroup,false);
        MyViewHolder cvh = new MyViewHolder(v);
        return cvh;
    }

    @Override
    public void onBindViewHolder(@NonNull MyViewHolder holder, int position) {
        MessageList currentItem = mElementList.get(position);
        holder.TextName.setText(currentItem.geteName());
        holder.TextDate.setText(currentItem.geteDate());
        holder.TextMessage.setText(currentItem.geteMessage());

        holder.cardView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                listener.onItemClicked(currentItem);
            }

        });
    }

    @Override
    public int getItemCount() {
        return mElementList.size();
    }

    public MessageList getMessageAt(int position){
        return mElementList.get(position);
    }

    public static class MyViewHolder extends RecyclerView.ViewHolder{

        public TextView TextName, TextDate,TextMessage;
        public CardView cardView;

        public MyViewHolder(@NonNull View itemView) {
            super(itemView);
            TextName = itemView.findViewById(R.id.id_nick);
            TextDate = itemView.findViewById(R.id.id_date);
            TextMessage = itemView.findViewById(R.id.id_message);
            cardView = itemView.findViewById(R.id.id_messageList);
        }
    }
}