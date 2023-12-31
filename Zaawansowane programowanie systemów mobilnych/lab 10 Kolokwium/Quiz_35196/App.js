// App.js

import React, { useRef, useState, useEffect } from 'react';
import { ScrollView, Button, Text, View, Image, TouchableOpacity } from 'react-native';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { NavigationContainer } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage'; 
import WelcomeScreen from './screens/WelcomeScreen'; 
import HomePageScreen from './screens/HomePageScreen';
import ResultsScreen from './screens/ResultsScreen';
import QuizEndScreen from './screens/QuizEndScreen';
import QuizScreen from './screens/QuizScreen';
import styles from './styles/DrawerStyle';

const Drawer = createDrawerNavigator();

const fetchTests = async (setTestsList, setIsOnline, setIsRefreshing) => {
  try {
    const response = await fetch('https://tgryl.pl/quiz/tests');
    const data = await response.json();
    const shuffledTests = [...data].sort(() => Math.random() - 0.5);
    
    const testsWithResults = [...shuffledTests, { resultsItem: true }];

    const testIds = testsWithResults.filter(item => !item.resultsItem).map(item => item.id);
    await AsyncStorage.setItem('testIds', JSON.stringify(testIds));

    const testsDataPromises = testsWithResults.map(async (test) => {
      if (!test.resultsItem) {
        const testResponse = await fetch(`https://tgryl.pl/quiz/test/${test.id}`);
        const testData = await testResponse.json();
        await AsyncStorage.setItem(`testData_${test.id}`, JSON.stringify(testData));
      }
    });
    await Promise.all(testsDataPromises);
    await AsyncStorage.setItem('tests', JSON.stringify(testsWithResults));
    console.log("Testy pobrane")
    setTestsList(testsWithResults);
    setIsOnline(true);
  } catch (error) {
    console.error('Error fetching tests:', error);
    setIsOnline(false);
  } finally {
    setIsRefreshing(false);
  }
};

const DrawerContent = ({ navigation }) => {
  const [testsList, setTestsList] = useState([]);
  const [isOnline, setIsOnline] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRandomTest = () => {
    const shuffledTests = shuffleTests();
    const randomTest = shuffledTests[0];

    if (randomTest) {
      navigation.navigate('Test', {
        testId: randomTest.id,
        titleTest: randomTest.name,
        typeTest: randomTest.type,
      });
    }
  };

  const handleSaveTestsToStorage = async () => {
    setIsRefreshing(true);
    await fetchTests(setTestsList, setIsOnline, setIsRefreshing);
  };

  useEffect(() => {
    const fetchTests = async () => {
      try {
        const response = await fetch('https://tgryl.pl/quiz/tests');
        const data = await response.json();
        setTestsList(data);
      } catch (error) {
        console.error('Error fetching tests:', error);
      }
    };

    fetchTests();
  }, []);

  const shuffleTests = () => {
    return testsList.sort(() => Math.random() - 0.5);
  };

  const renderTestButtons = () => {
    const shuffledTests = shuffleTests();
    const testIds = shuffledTests.filter(item => !item.resultsItem).map(item => item.id);
  
    return testIds.map((testId) => {
      const test = shuffledTests.find(item => item.id === testId);
  
      return (
        <TouchableOpacity
          key={testId}
          onPress={() =>
            navigation.navigate('Test', {
              testId: test.id,
              titleTest: test.name,
              typeTest: test.type,
            })
          }
        >
          <View style={styles.drawerButton}>
            <Text style={styles.drawerButtonText}>{test.name}</Text>
          </View>
        </TouchableOpacity>
      );
    });
  };
  
  return (
    <ScrollView contentContainerStyle={styles.scrollContainer}>
      <View style={[styles.containerDrawer, styles.navigationContainer]}>
        <Text style={styles.drawerTitle}>Quiz App</Text>
        <Image source={require('./assets/icon_choose.png')} style={styles.drawerIcon} />
        <View style={styles.buttonContainer}>
          <Button title="Home Page" onPress={() => navigation.navigate('Home Page')} color="#808080" />
          <View style={styles.buttonSpacer} />
          <Button title="Results" onPress={() => navigation.navigate('Results')} color="#808080" />
          <TouchableOpacity onPress={handleRandomTest}>
          <View style={styles.drawerDrawerButton}>
            <Text style={styles.drawerRandomButtonText}>RANDOM TEST</Text>
          </View>
        </TouchableOpacity>
        <TouchableOpacity onPress={handleSaveTestsToStorage}>
          <View style={styles.drawerDrawerButton}>
            <Text style={styles.drawerRandomButtonText}>SAVE TESTS TO STORAGE</Text>
          </View>
        </TouchableOpacity>
          <Text style={styles.divider}></Text>
          {renderTestButtons()}
      </View>
      </View>
    </ScrollView>
  );
};

const App = () => {
  const drawer = useRef(null);
  const [showWelcome, setShowWelcome] = useState(false);

  useEffect(() => {
    const checkRegulationAccepted = async () => {
      const isRegulationAccepted = await AsyncStorage.getItem('isRegulationAccepted');
      setShowWelcome(isRegulationAccepted !== 'true');
    };

    checkRegulationAccepted();
  }, []);

  const handleRegulationAccepted = async () => {
    await AsyncStorage.setItem('isRegulationAccepted', 'true');
    setShowWelcome(false);
  };

  return (
    <NavigationContainer>
      {showWelcome ? (
        <WelcomeScreen onRegulationAccepted={handleRegulationAccepted} />
      ) : (
        <Drawer.Navigator drawerContent={(props) => <DrawerContent {...props} />}>
          <Drawer.Screen name="Home Page" component={HomePageScreen} />
          <Drawer.Screen name="Results" component={ResultsScreen} />
          <Drawer.Screen name="Test" component={QuizScreen} />
          <Drawer.Screen name="Quiz Completed" component={QuizEndScreen} />
        </Drawer.Navigator>
      )}
    </NavigationContainer>
  );
};

export default App;