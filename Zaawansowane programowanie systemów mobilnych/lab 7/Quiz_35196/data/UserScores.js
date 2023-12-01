// UserScore.js

class UserScores {
    constructor(nick, point, type, date) {
      this.nick = nick;
      this.point = point;
      this.type = type;
      this.date = date;
    }
  }

  const userScores = [
    new UserScores("God", "20/20", "test1", "21-11-1990"),
    new UserScores("Master", "18/20", "test1", "18-11-1995"),
    new UserScores("Soul", "16/20", "test4", "11-11-1985"),
    new UserScores("Half", "13/20", "test2", "15-11-2000"),
    new UserScores("Qwerty", "9/20", "test5", "15-11-1998"),
    new UserScores("xcz", "8/20", "test4", "15-11-1982"),
    new UserScores("xyz", "5/20", "test1", "15-11-1976"),
    new UserScores("nobek", "3/20", "test5", "15-11-1993"),
    new UserScores("Bad man", "1/20", "test2", "15-11-1988"),
    new UserScores("Giga Noob", "0/20", "test1", "15-11-1999"),
  ];

  export { UserScores, userScores };