package com.kdbk.fiszki.RecyclerView.Adaper;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.RecyclerView;
import com.kdbk.fiszki.RecyclerView.Model.ModelCategories;
import com.kdbk.fiszki.R;
import com.kdbk.fiszki.RecyclerView.SelectListener.SelectListenerCategories;
import java.util.ArrayList;

public class AdapterCategories extends RecyclerView.Adapter<AdapterCategories.MyViewHolder> {
    private ArrayList<ModelCategories> listCategories;
    private SelectListenerCategories listener;

    public AdapterCategories(ArrayList<ModelCategories> listCategories, SelectListenerCategories listener){
        this.listCategories = listCategories;
        this.listener = listener;
    }

    @NonNull
    @Override
    public MyViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.recycler_view_categories, parent, false);
        MyViewHolder myHolder = new MyViewHolder(v);
        return myHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull MyViewHolder holder, int position) {
        ModelCategories currentItem = listCategories.get(position);
        holder.nameCategory.setText(currentItem.getNameCategory());
        holder.imageCategory.setImageResource(currentItem.getImageResource());

        holder.cardView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int clickedPosition = holder.getAdapterPosition();
                listener.onItemClicked(AdapterCategories.this.listCategories.get(clickedPosition));
            }
        });
    }

    @Override
    public int getItemCount() {
        return listCategories.size();
    }

    public static class MyViewHolder extends RecyclerView.ViewHolder{
        public TextView nameCategory;
        public ImageView imageCategory;
        public CardView cardView;

        public MyViewHolder(@NonNull View itemView) {
            super(itemView);
            nameCategory = itemView.findViewById(R.id.nameCategory);
            imageCategory = itemView.findViewById(R.id.imageCategory);
            cardView = itemView.findViewById(R.id.recycleCategory);
        }
    }
}
