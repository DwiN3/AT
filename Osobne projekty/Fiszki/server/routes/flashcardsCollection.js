const express = require('express');
const flashcardsCollectionController = require('../controllers/flashcardsCollection');
const isAuth = require('../middleware/auth');

const router = express.Router();

router.get('/', isAuth, flashcardsCollectionController.getCollections);
router.get('/:collectionName', isAuth, flashcardsCollectionController.getCollection);
router.delete('/:collectionName', isAuth, flashcardsCollectionController.deleteCollection);
router.get('/category/:category', isAuth, flashcardsCollectionController.getCategory);

module.exports = router;