module.exports = (error, req, res, next) => {

    console.log(error.statusCode);
    const status = error.statusCode || 500;
    const message = error.message;
    res.status(status).json({ message : message });

}