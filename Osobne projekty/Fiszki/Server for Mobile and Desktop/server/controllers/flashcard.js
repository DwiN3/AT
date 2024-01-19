const Flashcard = require('../models/flashcard');
const FlashcardsCollection = require('../models/flashcardsCollection');
const { validationResult } = require('express-validator');
const flashcard = require('../models/flashcard');

exports.addFlashCard = async(req, res, next) => {

    const errors = validationResult(req);

    if(!errors.isEmpty()){
        const error = new Error('Wrong data!');
        error.statusCode = 400;
        return next(error);
    }

    const collectionName = req.params.collectionName;
    const language = req.body.language;
    const category = req.body.category;
    const word = req.body.word;
    const translatedWord = req.body.translatedWord;
    const example = req.body.example;
    const translatedExample = req.body.translatedExample;
    req.file ? imgPath = req.file.path : imgPath = null;
    const author = req.user;
    
    try{
        const flashcardExist = await Flashcard.findOne({author : author, word : word, translatedWord : translatedWord, set : collectionName});
        if(flashcardExist){
            const error = new Error('This flashcard already exist in this set!');
            error.statusCode = 400;
            throw(error);
        }
        const newFlashcard = new Flashcard({
            set : collectionName,
            language : language,
            category : category,
            word : word,
            translatedWord : translatedWord,
            example : example,
            translatedExample : translatedExample,
            imgPath : imgPath,
            author : author
        })
        let collectionExist = await FlashcardsCollection.findOne({author: author, collectionName : collectionName});
        if(!collectionExist){
            collectionExist = await new FlashcardsCollection({
                collectionName : collectionName,
                author : author,
                flashcards : [newFlashcard],
                topResult : 0
            })
        }
        else collectionExist.flashcards.push(newFlashcard);
        await collectionExist.save();
        await newFlashcard.save();
    } catch(error){
        if(!error.statusCode) error.statusCode = 500;
        return next(error);
    }
    res.status(201).json('Flashcard added succesful!')
}

exports.deleteFlashcard = async(req, res, next) => {

    const flashcardsId = req.params.flashcardsId;
    try{
        const flashcard = await Flashcard.findOne({ author : req.user, _id : flashcardsId });
        if(!flashcard){
            const error = new Error('Flashcard doesnt exist!');
            error.statusCode = 400;
            throw (error);
        }
        const collection = await FlashcardsCollection.findOne({ author : req.user, collectionName : flashcard.set })    
        collection.flashcards = collection.flashcards.filter(flashcard => flashcard.toString() !== flashcardsId);
        if(collection.flashcards.length > 0) await collection.save()
        else await FlashcardsCollection.deleteOne({ _id : collection._id });
        await Flashcard.deleteOne({ _id : flashcardsId })
    } catch(error){
        if(!error.statusCode) error.statusCode = 500;
        return next(error);
    }
    return res.status(200).json('Flashcard deleted succesful!');

}

exports.editFlashcard = async(req, res, next) => {

    const errors = validationResult(req);

    if(!errors.isEmpty()){
        const error = new Error('Wrong data!');
        error.statusCode = 400;
        return next(error);
    }

    const flashcardsId = req.params.flashcardsId;
    try{
        const flashcard = await Flashcard.findOne({ author : req.user, _id : flashcardsId});
        if(!flashcard){
            const error = new Error('Flashcard doesnt exist!');
            error.statusCode = 400;
            throw (error);
        }
        flashcard.word = req.body.word;
        flashcard.translatedWord = req.body.translatedWord;
        flashcard.example = req.body.example;
        flashcard.translatedExample = req.body.translatedExample;
        await flashcard.save();
    } catch(error){
        if(!error.statusCode) error.statusCode = 500;
        return next(error);
    }
    return res.status(200).json('Flashcard edited succesful!');

}

exports.getFlashcard = async(req, res, next) => {

    const flashcardsId = req.params.flashcardsId;
    let flashcard
    try{
        flashcard = await Flashcard.findOne({author : req.user, _id : flashcardsId});
        if(!flashcard){
            const error = new Error('Flashcard doesnt exist!');
            error.statusCode = 400;
            throw (error);
        }
    }catch(error){
        if(!error.statusCode) error.statusCode = 500;
        return next(error);
    }
    return res.status(200).json(flashcard);

}