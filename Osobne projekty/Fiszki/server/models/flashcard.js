const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const flashcardSchema = new Schema({

    set : {
        type : String,
        required : true
    },

    language : {
        type : String,
        required : true
    },

    category : {
        type : String,
        required : true
    },

    word : {
        type : String,
        required : true
    },

    translatedWord : {
        type : String,
        required : true
    },

    example : {
        type : String,
        required : true
    },

    translatedExample : {
        type : String,
        reuired : true
    },

    imgPath : {
        type : String,
        required : false
    },

    author : {
        type : String,
        required : true
    }

});

module.exports = mongoose.model('Flashcard', flashcardSchema, 'flashcards');