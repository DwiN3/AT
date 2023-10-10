package com.kdbk.fiszki.RecyclerView.Adaper;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.RecyclerView;
import com.kdbk.fiszki.RecyclerView.Model.ModelKits;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.RecyclerView.SelectListener.SelectListenerKits;
import java.util.ArrayList;

public class AdapterKits extends RecyclerView.Adapter<AdapterKits.MyViewHolder> {
    private ArrayList<ModelKits> listKits;
    private SelectListenerKits listener;
    private static int infoLayout;

    public AdapterKits(ArrayList<ModelKits> listKits, SelectListenerKits listener, int infoLayout){
        this.listKits = listKits;
        this.listener = listener;
        this.infoLayout = infoLayout;
    }

    @NonNull
    @Override
    public MyViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(infoLayout, parent, false);
        MyViewHolder myHolder = new MyViewHolder(v);
        return myHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull MyViewHolder holder, int position) {
        ModelKits currentItem = listKits.get(position);
        holder.numberKit.setText(currentItem.getNameKit());
        holder.textTEXTflashcards.setText(currentItem.getTEXT());
        holder.numberOfCards.setText(String.valueOf(currentItem.getNumberOfCards()));

        holder.cardView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int clickedPosition = holder.getAdapterPosition();
                listener.onItemClicked(AdapterKits.this.listKits.get(clickedPosition));
            }
        });

    }

    @Override
    public int getItemCount() {
        return listKits.size();
    }

    public static class MyViewHolder extends RecyclerView.ViewHolder{
        public TextView numberKit, textTEXTflashcards, numberOfCards;
        public CardView cardView;
        public MyViewHolder(@NonNull View itemView) {
            super(itemView);
            if(infoLayout == R.layout.recycler_view_kits){
                numberKit = itemView.findViewById(R.id.textNumberKit);
                textTEXTflashcards = itemView.findViewById(R.id.TextTEXTflashcards);
                numberOfCards = itemView.findViewById(R.id.textNumberOfCards);
                cardView = itemView.findViewById(R.id.recycleKits);
            } else if (infoLayout == R.layout.recycler_view_kits_small) {
                numberKit = itemView.findViewById(R.id.textNumberKitSmall);
                textTEXTflashcards = itemView.findViewById(R.id.TextTEXTflashcardsSmall);
                numberOfCards = itemView.findViewById(R.id.textNumberOfCardsSmall);
                cardView = itemView.findViewById(R.id.recycleKitsSmall);
            }
        }
    }
}
