package com.kdbk.fiszki.RecyclerView.Adaper;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.RecyclerView;
import com.kdbk.fiszki.RecyclerView.Model.ModelShowKitsEdit;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.RecyclerView.SelectListener.SelectListenerShowKitsEdit;
import java.util.ArrayList;

public class AdapterShowKitsEdit extends RecyclerView.Adapter<AdapterShowKitsEdit.MyViewHolder> {

    private ArrayList<ModelShowKitsEdit> listKits;
    private SelectListenerShowKitsEdit listener;

    public AdapterShowKitsEdit(ArrayList<ModelShowKitsEdit> listKits, SelectListenerShowKitsEdit listener){
        this.listKits = listKits;
        this.listener = listener;
    }

    @NonNull
    @Override
    public AdapterShowKitsEdit.MyViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.recycler_view_show_kits, parent, false);
        AdapterShowKitsEdit.MyViewHolder myHolder = new AdapterShowKitsEdit.MyViewHolder(v);
        return myHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull AdapterShowKitsEdit.MyViewHolder holder, int position) {
        ModelShowKitsEdit currentItem = listKits.get(position);

        holder.cardView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int clickedPosition = holder.getAdapterPosition();
                listener.onItemClicked(AdapterShowKitsEdit.this.listKits.get(clickedPosition));
            }
        });
        holder.textWord.setText(currentItem.getWord());
        holder.textTranslateWord.setText(currentItem.getTranslateWord());
        holder.textSentens.setText(currentItem.getSentens());
        holder.textSentensTranslate.setText(currentItem.getSentensTranslate());
    }

    @Override
    public int getItemCount() {
        return listKits.size();
    }

    public static class MyViewHolder extends RecyclerView.ViewHolder{
        public CardView cardView;
        TextView textWord, textTranslateWord,textSentens, textSentensTranslate;

        public MyViewHolder(@NonNull View itemView) {
            super(itemView);
            textWord = itemView.findViewById(R.id.textViewShowKits1);
            textTranslateWord = itemView.findViewById(R.id.textViewShowKits2);
            textSentens = itemView.findViewById(R.id.textViewShowKits3);
            textSentensTranslate = itemView.findViewById(R.id.textViewShowKits4);
            cardView = itemView.findViewById(R.id.recycleShowFlashcard);
        }
    }
}
