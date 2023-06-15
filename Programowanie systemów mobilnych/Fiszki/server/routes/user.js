const express = require('express');
const userController = require('../controllers/user');
const { body } = require('express-validator');
const isAuth = require('../middleware/auth');

const router = express.Router();

router.post('/sing-up', 
    [
        body('email')
        .isEmail()
        .isLength({min : 10}),
        body('password')
            .isAlphanumeric()
            .isLength({ min : 8 }),
        body('repeatedPassword')
            .isAlphanumeric()
            .isLength({ min : 8 }),
        body('nick')
            .isAlphanumeric()
            .isLength({ min : 5 })
    ],
    userController.singUp);

router.post('/login', 
    [
        body('nick')
            .isAlphanumeric()
            .isLength({ min : 5}),
        body('password')
            .isAlphanumeric()
            .isLength({ min : 8 })
    ],
    userController.login);

router.get('/users-level', isAuth, userController.getUsersLevel);

router.put('/users-level', isAuth, 
    [
        body('result')
            .isInt({min : 0})
    ],
    userController.levelUp);

router.put('/password-reset', 
    [
        body('email')
            .isEmail()
            .isLength({min : 10}),
        body('password')
            .isAlphanumeric()
            .isLength({ min : 8 }),
        body('repeatedPassword')
            .isAlphanumeric()
            .isLength({ min : 8 }),
    ],
    userController.changePassword);

    router.post('/nick-remind', 
    [
        body('email')
            .isEmail()
            .isLength({min : 10}),
    ],
    userController.remindNick);

module.exports = router;

