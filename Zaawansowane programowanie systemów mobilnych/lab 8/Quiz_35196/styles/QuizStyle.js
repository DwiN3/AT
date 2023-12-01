// QuizStyle.js

import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  containerQuiz: {
    flex: 1,
    flexDirection: 'column',
  },
  textContainer: {
    //backgroundColor: 'blue',
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 10,
    height: 50, 
  },
  questionNumbersText: {
    marginTop: 25,
    marginLeft: 25,
    fontSize: 18,
  },
  timeText: {
    marginTop: 25,
    marginRight: 25,
    fontSize: 18,
  },
  progressBarContainer:{
    marginTop: 20,
    width: '80%',
    borderWidth: 2,
    borderBlockColor: 'black',
    borderRadius: 30,
    alignSelf: 'center',
  },
  questionContainer: {
    marginTop: 5,
    //backgroundColor: 'pink',
    height: 150, 
  },
  questionText: {
    marginTop: 18,
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  descriptionContainer: {
    height: 150,
    //backgroundColor: 'red',
    marginTop: -50,  
  },
  descriptionText: {
    fontSize: 13,
    textAlign: 'justify',
    marginLeft: 10,
    marginRight: 10,
  },
  answersContainer: {
    borderWidth: 2,
    borderColor: 'black',
    borderRadius: 3,
    marginTop: 20,
    padding: 25,
    marginLeft: 5,
    marginRight: 5,
  },
  answersButton: {
    backgroundColor: '#808080',
    borderWidth: 2,
    borderColor: 'black',
    borderRadius: 3,
    padding: 10,
    margin: 5,
    alignItems: 'center',
    height: 45,
    width: 160,
  },
  answersButtonText: {
    fontSize: 15,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between', 
    flexWrap: 'wrap',
  },
});

export default styles;
