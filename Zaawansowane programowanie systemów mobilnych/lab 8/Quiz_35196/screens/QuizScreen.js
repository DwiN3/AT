// QuizScreen.js

import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, Alert } from 'react-native'; 
import { useFocusEffect, useRoute } from '@react-navigation/native';
import styles from '../styles/QuizStyle';

const QuizScreen = ({ navigation }) => {
  const [progress, setProgress] = useState(0);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [questionTime, setQuestionTime] = useState(0);
  const [shouldStartTimer, setShouldStartTimer] = useState(true);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [correctAnswers, setCorrectAnswers] = useState(0);
  const [resetQuizFlag, setResetQuizFlag] = useState(false); // New state variable
  const intervalRef = useRef(null);
  const prevTitleRef = useRef(null); 

  const route = useRoute();
  const { params } = route;
  const titleTest = params ? params.titleTest : null;
  const description = params ? params.description : null;
  const tasks = params ? params.tasks : [];

  useEffect(() => {
    if (titleTest && titleTest !== prevTitleRef.current) {
      navigation.setOptions({ title: titleTest });
      setResetQuizFlag(true);
      prevTitleRef.current = titleTest;
      resetQuiz();
    }
    setQuestionTime(tasks[currentQuestion]?.duration || 0);
    resetTimerFunction(); 
  }, [titleTest, currentQuestion, tasks, resetQuizFlag]);

  useEffect(() => {
    if (shouldStartTimer) {
      resetTimer();
      setShouldStartTimer(true);
    }
    return () => clearInterval(intervalRef.current);
  }, [questionTime, shouldStartTimer]);

  useFocusEffect(
    React.useCallback(() => {
      setShouldStartTimer(true);
      return () => {
        clearInterval(intervalRef.current);
      };
    }, [])
  );

  const resetTimerFunction = () => {
    resetTimer();
    return () => {
      clearInterval(intervalRef.current);
    };
  };

  const startTimer = () => {
    intervalRef.current = setInterval(() => {
      setTimeElapsed((prevTime) => {
        const newTime = prevTime + 1;
        setProgress((prevProgress) => newTime / questionTime);

        if (newTime === questionTime) {
          clearInterval(intervalRef.current);
          console.log('Koniec czasu');
          handleAnswer(null);
        }

        return newTime;
      });
    }, 1000);
  };

  const resetTimer = () => {
    clearInterval(intervalRef.current);
    setTimeElapsed(0);
    setProgress(0);
    startTimer();
  };

  const resetQuiz = () => {
    setProgress(0);
    setTimeElapsed(0);
    setShouldStartTimer(true);
    setCurrentQuestion(0);
    setCorrectAnswers(0);
  };

  const handleAnswer = (selectedAnswer) => {
    clearInterval(intervalRef.current); 
    moveToNextQuestion(selectedAnswer);
  };

  let globalCorrectAnswers = 0;
  const alertEnd = () => {
    clearInterval(intervalRef.current); 

    Alert.alert(
      'Quiz Finish',
      'Congratulations! You have completed the quiz.',
      [
        {
          text: 'Go to Results',
          onPress: () => {
            navigation.navigate('Quiz Completed', { 
              textTitle: titleTest,
              globalCorrectAnswers: globalCorrectAnswers,
              totalQuestions: tasks.length });
          },
        },
      ],
    );
  };

  const moveToNextQuestion = (selectedAnswer) => {
    const isCorrect = tasks[currentQuestion]?.answers[selectedAnswer]?.isCorrect || false;
  
    setCorrectAnswers((prevCorrectAnswers) => {
      const newCorrectAnswers = isCorrect ? prevCorrectAnswers + 1 : prevCorrectAnswers;
      //console.log('Wartość zmiennej isCorrect:', isCorrect, newCorrectAnswers);
      globalCorrectAnswers = newCorrectAnswers;
  
      return newCorrectAnswers;
    });
  
    if (currentQuestion + 1 < tasks.length) {
      setCurrentQuestion((prevQuestion) => prevQuestion + 1);
      setQuestionTime(tasks[currentQuestion + 1]?.duration || 0);
      resetTimerFunction();
    } else {
      alertEnd();
    }
  };

  return (
    <View style={styles.containerQuiz}>
      <View style={styles.textContainer}>
        <Text style={styles.questionNumbersText}>{`Question ${currentQuestion + 1} of ${tasks.length}`}</Text>
        <Text style={styles.timeText}>Time: {questionTime - timeElapsed} sec</Text>
      </View>
      <View style={styles.progressBarContainer}>
        <View style={{ backgroundColor: 'yellow', height: 10, width: `${progress * 100}%` }} />
      </View>
      <View style={styles.questionContainer}>
        <Text style={styles.questionText}>{tasks[currentQuestion]?.question}</Text>
      </View>
      <View style={styles.descriptionContainer}>
        <Text style={styles.descriptionText}>{description}</Text>
      </View>
      <View style={styles.answersContainer}>
        <View style={styles.buttonRow}>
          {tasks[currentQuestion]?.answers.map((answer, index) => (
            <TouchableOpacity key={index} style={styles.answersButton} onPress={() => handleAnswer(index)}>
              <Text style={styles.answersButtonText}>{`${answer.content}`}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    </View>
  );
};

export default QuizScreen;
