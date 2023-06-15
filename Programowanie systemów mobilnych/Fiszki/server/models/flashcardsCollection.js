const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const flashcardsCollectionSchema = new Schema({

    collectionName : {
        type : String,
        required : true
    },

    author : {
        type : String,
        required : true
    },

    flashcards : [{
        type : Schema.Types.ObjectId,
        ref : 'Flashcard'
    }],

    topResult : {
        type : Number,
        required : true,
    } 

});

module.exports = mongoose.model('FlashcardCollection', flashcardsCollectionSchema, 'flashcardCollections');