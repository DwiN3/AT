const express = require('express');
const questions = require('./questions').questions;
const { serverPort } = require('./config');

const app = express();

app.get('/api/questions', (request, response) => {
    response.send(questions);
});

app.get('/api/questions/:id', (requset, response) => {
    const id = requset.params.id;
    const question = questions[id];

    if (!question) {
        return response.status(404).send('No question with this id');
    }

    response.send(question);
});

app.post('/api/questions/:id', (request, response) => {
    const id = request.params.id;
    const question = request.body;

    if (!question) {
        return response.status(400).send('Question data is required');
    }

    if (!questions[id]) {
        return response.status(404).send('No question with this id');
    }

    questions[id].answers.push(question); 
    response.status(201).send('Question added successfully');
});

app.listen(serverPort, function () {
console.info(`Server is running at port 3000`);
});