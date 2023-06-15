const express = require('express');
const flashcardController = require('../controllers/flashcard');
const { body } = require('express-validator');
const isAuth = require('../middleware/auth');

const router = express.Router();

router.get('/:flashcardsId', isAuth, flashcardController.getFlashcard);
router.post('/:collectionName', isAuth,  
    [
        body('language')
            .isIn('english'),
        body('category')
            .isIn(['inne', 'dom', 'zakupy', 'praca', 'zdrowie', 'czlowiek', 'turystyka', 'jedzenie', 'edukacja', 'zwierzeta' ]),
        body('word')
            .isLength({ min : 2 }),
        body('translatedWord')
            .isLength({ min : 2 }),
        body('example')
            .isLength({ min : 3}),
        body('translatedExample')
            .isLength({ min : 3}),
    ],
    flashcardController.addFlashCard);
router.delete('/:flashcardsId', isAuth, flashcardController.deleteFlashcard);
router.put('/:flashcardsId', isAuth, 
    [
        body('word')
            .isLength({ min : 2 }),
        body('translatedWord')
            .isLength({ min : 2 }),
        body('example')
            .isLength({ min : 3}),
        body('translatedExample')
            .isLength({ min : 3}),
    ],
    flashcardController.editFlashcard);
module.exports = router;


