// QuizScreen.js

import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { useFocusEffect, useRoute } from '@react-navigation/native';
import styles from '../styles/QuizStyle';

const QuizScreen = ({ navigation }) => {
  const [progress, setProgress] = useState(0);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [questionTime, setQuestionTime] = useState(15);
  const [shouldStartTimer, setShouldStartTimer] = useState(true);
  const intervalRef = useRef(null);

  const route = useRoute();
  const { params } = route;
  const titleTest = params ? params.titleTest : null;

  useEffect(() => {
    if (titleTest) {
      navigation.setOptions({ title: titleTest });
      resetTimer();
      return () => {
        clearInterval(intervalRef.current);
      };
    }
  }, [titleTest]);

  useEffect(() => {
    if (shouldStartTimer) {
      resetTimer();
      setShouldStartTimer(false);
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

  const startTimer = () => {
    intervalRef.current = setInterval(() => {
      setTimeElapsed((prevTime) => {
        const newTime = prevTime + 1;
        setProgress((prevProgress) => newTime / questionTime);

        if (newTime === questionTime) {
          clearInterval(intervalRef.current);
          console.log('Koniec czasu');
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

  return (
    <View style={styles.containerQuiz}>
      <View style={styles.textContainer}>
        <Text style={styles.questionNumbersText}>Question 3 of 10</Text>
        <Text style={styles.timeText}>Time: {questionTime - timeElapsed} sec</Text>
      </View>
      <View style={styles.progressBarContainer}>
        <View style={{ backgroundColor: 'yellow', height: 10, width: `${progress * 100}%` }} />
      </View>
      <View style={styles.questionContainer}>
        <Text style={styles.questionText}>
          This is some example of a long question to fill the content?
        </Text>
        <Text style={styles.descriptionText}>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla quam velit. vulputate eu
          pharetra nec, mattis ac neque. Duis vulputate commod...
        </Text>
      </View>
      <View style={styles.answersContainer}>
        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.answersButton}>
            <Text style={styles.answersButtonText}>Answer A</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.answersButton}>
            <Text style={styles.answersButtonText}>Answer B</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.answersButton}>
            <Text style={styles.answersButtonText}>Answer C</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.answersButton}>
            <Text style={styles.answersButtonText}>Answer D</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

export default QuizScreen;
