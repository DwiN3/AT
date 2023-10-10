const Flashcardcollection = require('../models/flashcardsCollection');
const Flashcard = require('../models/flashcard'); 
const randomIndex = require('../middleware/randomIndex');
const flashcard = require('../models/flashcard');

exports.getCollections = async(req, res, next) => {

    let flashcards;

    try{
        flashcards = await Flashcardcollection.find({ author : req.user})
    } catch(error){
        if(!error.statusCode) error.statusCode === 500;
        return next(error);
    }
    res.status(200).json(flashcards);

}

exports.getCollection = async(req, res, next) => {

    const collectionName = req.params.collectionName;

    try{
        collection = await Flashcardcollection.findOne({ author : req.user, collectionName : collectionName}).populate('flashcards')
        if(!collection){
            const error = new Error("Collection doesn't exist");
            error.statusCode = 400;
            throw(error);
        }
    } catch(error){
        if(!error.statusCode) error.statusCode === 500;
        return next(error);
    }
    res.status(200).json(collection);

}

exports.deleteCollection = async(req, res, next) => {

    const collectionName = req.params.collectionName;

    try{
        const collection = await Flashcardcollection.findOne({ author : req.user, collectionName : collectionName })
        if(!collection){
            const error = new Error('Collection doesnt exist!');
            error.statusCode = 400;
            throw (error);
        }
        await Flashcardcollection.deleteOne({ author : req.user, collectionName : collectionName});
        await Flashcard.deleteMany({ author : req.user, set : collectionName });
    } catch(error){
        if(!error.statusCode) error.statusCode = 500;
        return next(error);
    }
    return res.status(200).json('Deleted succesful!');

}

exports.getCategory = async(req, res, next) => {

    const flashcards = [];
    const category = req.params.category
    try{
        const size = await Flashcard
            .countDocuments({category : category});
        const randomArray = randomIndex(size);
        for(const index of randomArray){
            const flashcard = await Flashcard
            .find({category : category })
            .limit(1)
            .skip(index - 1)
            .select('word translatedWord example translatedExample');
            flashcards.push(flashcard);
        };
        if(flashcards.length === 0){
            const error = new Error('Category does not exist!');
            error.statusCode = 400;
            throw (error);
        }
    } catch(error){
        if(!error.statusCode) error.statusCode = 500;
        return next(error);
    }
    return res.status(200).json(flashcards);
}