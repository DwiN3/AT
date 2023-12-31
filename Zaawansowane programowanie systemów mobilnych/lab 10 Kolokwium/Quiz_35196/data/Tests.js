// Tests.js

class Tests {
  constructor(title, tags, description, tasks) {
    this.titleTest = title;
    this.tags = tags;
    this.description = description;
    this.tasks = tasks;
  }
}

const tasks = [
  {
    "question": "",
    "answers": [
      {
        "content": "",
        "isCorrect": true
      },
      {
        "content": "",
        "isCorrect": false
      },
      {
        "content": "",
        "isCorrect": false
      },
      {
        "content": "",
        "isCorrect": false
      }
    ],
    "duration": 0
  },
];


const TestsList = [
  new Tests("Title Test", ['#Tag1', '#Tag2'], 'Description', tasks),
];

const savedTestList = [

];

const savedIDTest = [

];

const savedTasksList = [

];

export { Tests, TestsList, savedTestList, savedIDTest, savedTasksList };