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

const DrawerContent = ({ navigation }) => {
  const [testsList, setTestsList] = useState([]);

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

  const renderTestButtons = () => {
    return testsList.map((test) => (
      <TouchableOpacity
        key={test.id}
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
    ));
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