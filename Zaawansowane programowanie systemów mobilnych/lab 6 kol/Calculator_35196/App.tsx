import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Text, StyleSheet, Dimensions } from 'react-native';
import SplashScreen from 'react-native-splash-screen'
import { clearDisplay, calculateResult, power, sqrt, procent, minus_plus, displayPi, ln, log10, reciprocal, displayEuler, scientificNotation, powerOfTen, power2nd, expPower, degreesToRadians, random, addToMemory, subtractFromMemory, recallFromMemory, clearMemory, factorial, trigFunction} from './CalculatorFunction';

const App = () => {
  const [displayText, setDisplayText] = useState('0');
  const [memory, setMemory] = useState(0);
  const [orientation, setOrientation] = useState(
    Dimensions.get('window').width > Dimensions.get('window').height ? 'landscape' : 'portrait'
  );

  const handleButtonPress = (buttonValue) => {
    if (displayText === '0' && /[0-9]/.test(buttonValue)) {
      setDisplayText(buttonValue);
    } else {
      setDisplayText((prevText) => {
        if (prevText === '0' && buttonValue === '(') {
          return buttonValue;
        } else {
          return prevText + buttonValue;
        }
      });
    }
  };
  
  const handleOrientationChange = () => {
    const { width, height } = Dimensions.get('window');
    if (width > height) {
      setOrientation('landscape');
    } else {
      setOrientation('portrait');
    }
  };

  useEffect(() => {
    if(Platform.OS=== 'android') SplashScreen.hide();
    
    Dimensions.addEventListener('change', handleOrientationChange);

    return () => {
      Dimensions.removeEventListener('change', handleOrientationChange);
    };
  }, []);

  const buttonsPortrait = [
    { title: 'AC', onPress: () => clearDisplay(setDisplayText), style: styles.buttonMore },
    { title: '', style: styles.buttonMore },
    { title: '', style: styles.buttonMore },
    { title: '÷', onPress: () => handleButtonPress('/'), style: styles.buttonOperator },
    { title: '7', onPress: () => handleButtonPress('7'), style: styles.buttonNumber },
    { title: '8', onPress: () => handleButtonPress('8'), style: styles.buttonNumber },
    { title: '9', onPress: () => handleButtonPress('9'), style: styles.buttonNumber },
    { title: '×', onPress: () => handleButtonPress('*'), style: styles.buttonOperator },
    { title: '4', onPress: () => handleButtonPress('4'), style: styles.buttonNumber },
    { title: '5', onPress: () => handleButtonPress('5'), style: styles.buttonNumber },
    { title: '6', onPress: () => handleButtonPress('6'), style: styles.buttonNumber },
    { title: '-', onPress: () => handleButtonPress('-'), style: styles.buttonOperator },
    { title: '1', onPress: () => handleButtonPress('1'), style: styles.buttonNumber },
    { title: '2', onPress: () => handleButtonPress('2'), style: styles.buttonNumber },
    { title: '3', onPress: () => handleButtonPress('3'), style: styles.buttonNumber },
    { title: '+', onPress: () => handleButtonPress('+'), style: styles.buttonOperator },
    { title: '0', onPress: () => handleButtonPress('0'), style: [ styles.buttonNumber, { flex: 2 }]},
    { title: '.', onPress: () => handleButtonPress('.'), style: styles.buttonNumber },
    { title: '=', onPress: () => calculateResult(displayText, setDisplayText), style: styles.buttonOperator },
  ];

  const buttonsLand = [
    { title: '(', onPress: () => handleButtonPress('('), style: styles.buttonMore },
    { title: ')', onPress: () => handleButtonPress(')'), style: styles.buttonMore },
    { title: 'mc', onPress: () => clearMemory(setMemory), style: styles.buttonMore },
    { title: 'm+', onPress: () => addToMemory(displayText, setDisplayText, memory, setMemory), style: styles.buttonMore },
    { title: 'm-', onPress: () => subtractFromMemory(displayText, setDisplayText, memory, setMemory), style: styles.buttonMore },
    { title: 'mr', onPress: () => recallFromMemory(displayText, setDisplayText, memory), style: styles.buttonMore },
    { title: 'AC', onPress: () => clearDisplay(setDisplayText), style: styles.buttonMore },
    { title: '+/-', onPress: () => minus_plus(displayText, setDisplayText), style: styles.buttonMore },
    { title: '%', onPress: () => procent(displayText, setDisplayText), style: styles.buttonMore },
    { title: '÷', onPress: () => handleButtonPress('/'), style: styles.buttonMore },

    { title: '2nd', onPress: () => power2nd(displayText, setDisplayText), style: styles.buttonMore },
    { title: 'x²', onPress: () => power(displayText, setDisplayText,2), style: styles.buttonMore },
    { title: 'x³', onPress: () => power(displayText, setDisplayText,3), style: styles.buttonMore },
    { title: 'x^y', onPress: () => handleButtonPress('^'), style: styles.buttonMore },
    { title: 'e^x', onPress: () => expPower(displayText, setDisplayText), style: styles.buttonMore },
    { title: '10^x', onPress: () => powerOfTen(displayText, setDisplayText), style: styles.buttonMore },
    { title: '7', onPress: () => handleButtonPress('7'), style: styles.buttonNumber },
    { title: '8', onPress: () => handleButtonPress('8'), style: styles.buttonNumber },
    { title: '9', onPress: () => handleButtonPress('9'), style: styles.buttonNumber },
    { title: '×', onPress: () => handleButtonPress('*'), style: styles.buttonOperator },

    { title: '1/x', onPress: () => reciprocal(displayText, setDisplayText), style: styles.buttonMore },
    { title: '√x', onPress: () => sqrt(displayText, setDisplayText,'√'), style: styles.buttonMore },
    { title: '³√x', onPress: () => sqrt(displayText, setDisplayText,'³√'), style: styles.buttonMore },
    { title: 'y√x', onPress: () => handleButtonPress('√'), style: styles.buttonMore },
    { title: 'In', onPress: () => ln(displayText, setDisplayText), style: styles.buttonMore },
    { title: 'log10', onPress: () => log10(displayText, setDisplayText), style: styles.buttonMore },
    { title: '4', onPress: () => handleButtonPress('4'), style: styles.buttonNumber },
    { title: '5', onPress: () => handleButtonPress('5'), style: styles.buttonNumber },
    { title: '6', onPress: () => handleButtonPress('6'), style: styles.buttonNumber },
    { title: '-', onPress: () => handleButtonPress('-'), style: styles.buttonOperator },

    { title: 'x!', onPress: () => factorial(displayText, setDisplayText), style: styles.buttonMore },
    { title: 'sin', onPress: () => trigFunction(displayText, setDisplayText,'sin'), style: styles.buttonMore },
    { title: 'cos', onPress: () => trigFunction(displayText, setDisplayText,'cos'), style: styles.buttonMore },
    { title: 'tan', onPress: () => trigFunction(displayText, setDisplayText,'tan'), style: styles.buttonMore },
    { title: 'e', onPress: () => displayEuler(setDisplayText), style: styles.buttonMore },
    { title: 'EE', onPress: () => scientificNotation(displayText, setDisplayText), style: styles.buttonMore },
    { title: '1', onPress: () => handleButtonPress('1'), style: styles.buttonNumber },
    { title: '2', onPress: () => handleButtonPress('2'), style: styles.buttonNumber },
    { title: '3', onPress: () => handleButtonPress('3'), style: styles.buttonNumber },
    { title: '+', onPress: () => handleButtonPress('+'), style: styles.buttonOperator },

    { title: 'Rad', onPress: () => degreesToRadians(displayText, setDisplayText), style: styles.buttonMore },
    { title: 'sinh', onPress: () => trigFunction(displayText, setDisplayText,'sinh'), style: styles.buttonMore },
    { title: 'cosh', onPress: () => trigFunction(displayText, setDisplayText,'cosh'), style: styles.buttonMore },
    { title: 'tanh', onPress: () => trigFunction(displayText, setDisplayText,'tanh'), style: styles.buttonMore },
    { title: 'π', onPress: () => displayPi(setDisplayText), style: styles.buttonMore },
    { title: 'Rand', onPress: () => random(setDisplayText), style:[ styles.buttonMore, { flex:1 } ]},
    { title: '0', onPress: () => handleButtonPress('0'), style: [styles.buttonNumber, { flex: 2 }],},
    { title: '.', onPress: () => handleButtonPress('.'), style: styles.buttonNumber },
    { title: '=', onPress: () => calculateResult(displayText, setDisplayText), style: styles.buttonOperator },
  ];
  const buttonsToRender = orientation === 'landscape' ? buttonsLand : buttonsPortrait;
  const sliceCount = orientation === 'landscape' ? 10 : 4;
  const sizeFont = orientation === 'landscape'
  ? { ...styles.buttonText, fontSize: 15 }
  : { ...styles.buttonText, fontSize: 35 };

  return (
    <View style={styles.container}>
      <View style={styles.display}>
        <Text style={styles.displayText}>{displayText}</Text>
      </View>
      <View style={styles.buttons}>
        {[0, 1, 2, 3, 4].map((row) => (
          <View key={row} style={styles.row}>
            {buttonsToRender.slice(row * sliceCount, (row + 1) * sliceCount).map((button, index) => (
              <TouchableOpacity
                key={index}
                style={button.style}
                onPress={button.onPress}
              >
                <Text style={sizeFont}>{button.title}</Text>
              </TouchableOpacity>
            ))}
          </View>
        ))}
      </View>
    </View>
  );
};

const commonButtonStyle = {
  flex: 1,
  borderWidth: 1,
  borderColor: '#434343',
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#535457',
  },
  display: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'flex-end',
    backgroundColor: '#535457',
    padding: 10,
  },
  displayText: {
    fontSize: 65,
    color: 'white',
  },
  buttons: {
    flex: 4,
    flexDirection: 'column',
  },
  row: {
    flex: 2,
    flexDirection: 'row',
  },
  buttonText: {
    color: 'white',
    fontSize: 10,
    paddingVertical: 20,
    textAlign: 'center',
  },
  buttonMore: {
    ...commonButtonStyle,
    backgroundColor: '#646466',
  },
  buttonOperator: {
    ...commonButtonStyle,
    backgroundColor: 'orange',
  },
  buttonNumber: {
    ...commonButtonStyle,
    backgroundColor: '#7c7d7f',
  },
});

export default App;