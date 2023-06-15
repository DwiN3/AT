const jwt = require('jsonwebtoken');

module.exports = (req, res, next) => {
    const authHeader = req.get('Authorization');
    if(!authHeader){
        const error = new Error('Not authenticated!');
        error.statusCode = 401;
        return next(error);
    }
    const token = req.get('Authorization').split(' ')[1];
    let decodedToken;
    try{
        decodedToken = jwt.verify(token, 'flashcardsproject');
    } catch(error){
        error.message = 'Problem with serwer, try leter!'
        if(!statusCode) error.statusCode = 500;
        return next(error);
    }
    if(!decodedToken){
        const error = new Error('Not authenticated!');
        error.statusCode = 401;
        return next(error);
    }
    req.user = decodedToken.nick;
    next();
}